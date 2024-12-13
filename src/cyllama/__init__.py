import os
import sys
import datetime
from pathlib import Path

from typing import Union, Optional

Pathlike = Union[str, Path]

from . import cyllama as cy
from . import log


class Llama:
    """top-level api class for llamalib"""

    def __init__(
        self,
        model_path: Pathlike,
        n_predict: int = 512,
        n_ctx: int = 2048,
        disable_log: bool = True,
        n_threads: int = 4,
    ):
        self.model_path = Path(model_path)
        self.disable_log = disable_log
        self.log = log.config(self.__class__.__name__)
        if not self.model_path.exists():
            raise SystemExit(f"Provided model does not exist: {model_path}")

        self.params = cy.CommonParams()
        self.params.model = str(self.model_path)
        self.params.n_predict = n_predict
        self.params.n_ctx = n_ctx
        self.params.verbosity = -1
        self.params.cpuparams.n_threads = n_threads

        self.model: Optional[cy.LlamaModel] = None
        self.ctx: Optional[cy.LlamaContext] = None
        self.smplr: Optional[cy.LlamaSampler] = None

        self.chat_msgs: list[cy.CommonChatMsg] = []

        self.path_session = Path(self.params.path_prompt_cache)
        self.session_tokens = []
        self.add_bos: bool = False

        if self.disable_log:
            cy.log_set_verbosity(self.params.verbosity)

        cy.common_init()

        # run configuration checks
        self.check_params()

        # init LLM
        cy.llama_backend_init()
        cy.llama_numa_init(self.params.numa)

    def __del__(self):
        cy.llama_backend_free()

    def check_params(self):
        if self.params.logits_all:
            self.fail("please use the 'perplexity' tool for perplexity calculations")

        if self.params.embedding:
            self.fail("please use the 'embedding' tool for embedding calculations")

        if self.params.n_ctx != 0 and self.params.n_ctx < 8:
            self.log.warn("minimum context size is 8, using minimum size of 8.")
            self.params.n_ctx = 8

        if self.params.rope_freq_base != 0.0:
            self.log.warn(
                "changing RoPE frequency base to %g", self.params.rope_freq_base
            )

        if self.params.rope_freq_scale != 0.0:
            self.log.warn(
                "changing RoPE frequency base to %g", self.params.rope_freq_scale
            )

    def fail(self, msg, *args):
        """exits the program with an error msg."""
        self.log.critical(msg, *args)
        sys.exit(1)

    def attach_threadpool(self, ctx: cy.LlamaContext):
        # MOVE this as context is defined later
        tpp_batch = cy.GGMLThreadPoolParams.from_cpu_params(self.params.cpuparams_batch)
        tpp = cy.GGMLThreadPoolParams.from_cpu_params(self.params.cpuparams)

        cy.set_process_priority(self.params.cpuparams.priority)

        if tpp.match(tpp_batch):
            threadpool_batch = cy.GGMLThreadPool(tpp_batch)
            tpp.paused = True

            threadpool = cy.GGMLThreadPool(tpp)

            ctx.attach_threadpool(threadpool, threadpool_batch)

    def chat_add_and_format(self, chat_msgs: list[cy.CommonChatMsg], role: str, content: str) -> str:
        new_msg = cy.CommonChatMsg(role, content)
        formatted = cy.common_chat_format_single(self.model, self.chat_template, chat_msgs, new_msg, role == "user");
        chat_msgs.append(CommonChatMsg(role, content))
        self.log.debug("formatted: '%s'\n", formatted)
        return formatted

    def ask(self, prompt: str, n_predict: Optional[int] = None, n_ctx: Optional[int] = None):
        """prompt model"""

        self.params.prompt = prompt

        if n_predict:
            self.params.n_predict = n_predict
        if n_ctx:
            self.params.n_ctx = n_ctx

        # total length of the sequence including the prompt
        n_predict: int = self.params.n_predict

        # FIXME: PICK ONE, both ways work!
        if 0:
            # initialize the model
            model_params = cy.common_model_params_to_llama(self.params)
            self.model = cy.LlamaModel(path_model=self.params.model, params=model_params)

            # initialize the context
            ctx_params = cy.common_context_params_to_llama(self.params)
            self.ctx = cy.LlamaContext(model=self.model, params=ctx_params)
        else:
            # load the model and apply lora adapter, if any
            llama_init = cy.CommonInitResult(self.params)
            self.model = llama_init.model
            self.ctx = llama_init.context

        # attach threadpool
        self.attach_threadpool(self.ctx)

        # build sampler chain
        sparams = cy.LlamaSamplerChainParams()
        sparams.no_perf = False

        self.smplr = cy.LlamaSampler(sparams)
        self.smplr.add_greedy()

        # tokenize the prompt
        tokens_list: list[int] = cy.common_tokenize(self.ctx, self.params.prompt, True)
        n_ctx_train: int = self.model.n_ctx_train
        n_ctx: int = self.ctx.n_ctx

        if n_ctx > n_ctx_train:
            self.log.warn("model was trained on only %d context tokens (%d specified)", n_ctx_train, n_ctx)

        # print chat template example in conversation mode
        if self.params.conversation:
            if self.params.enable_chat_template:
                self.log.info("chat template example:\n%s\n", cy.common_chat_format_example(self.model, self.params.chat_template))
            else:
                self.log.info("in-suffix/prefix is specified, chat template will be disabled.")

        if not self.disable_log:
            # print system information
            self.log.info("\n%s\n", cy.common_params_get_system_info(self.params))

        if self.path_session.name: # is not empty
            self.log.info("attempting to load saved session from '%s'\n", self.path_session)
            if not self.path_session.exists():
                self.log.info("session file does not exist, will create.")
            elif self.path_session.read_text()=='':
                self.log.info("The session file is empty. A new session will be initialized.")
            else:
                # The file exists and is not empty
                self.session_tokens = self.ctx.load_state_file(path_session=self.path_session, max_n_tokens=n_ctx)
                self.log.info("loaded a session with prompt size of %d tokens", len(self.session_tokens));

        self.add_bos = self.model.add_bos_token()

        if not self.model.has_encoder():
            self.model.add_eos_token()

        self.log.debug("n_ctx: %d, add_bos: %d\n", n_ctx, self.add_bos)

        embd_inp = []

        if self.params.conversation and self.params.enable_chat_template and not self.params.prompt:
            # format the system prompt in conversation mode
            prompt = self.chat_add_and_format(self.chat_msgs, "system", self.params.prompt)
        else:
            prompt = self.params.prompt

        if self.params.interactive_first or not self.params.prompt or not self.session_tokens:
            self.log.debug("tokenize the prompt")
            embd_inp = cy.common_tokenize(self.ctx, prompt, True, True)
        else:
            self.log.debug("use session tokens")
            embd_inp = self.session_tokens

        self.log.debug('prompt: "%s"\n', prompt)
        self.log.debug("tokens: %s\n", cy.string_from_tokens(self.ctx, embd_inp))

        # Should not run without any tokens
        if not embd_inp:
            if self.add_bos:
                embd_inp.append(self.model.token_bos())
                self.log.warn("embd_inp was considered empty and bos was added: %s\n", cy.string_from_tokens(self.ctx, embd_inp))
            else:
                self.log.error("input is empty\n")
                raise SystemExit

        # Tokenize negative prompt
        if len(embd_inp) > n_ctx - 4:
            self.log.error("prompt is too long (%d tokens, max %d)\n", len(embd_inp), n_ctx - 4)
            raise SystemExit






        n_kv_req: int = len(tokens_list) + (n_predict - len(tokens_list))

        if not self.disable_log:
            self.log.info(
                "n_predict = %d, n_ctx = %d, n_kv_req = %d"
                % (n_predict, n_ctx, n_kv_req)
            )

        if n_kv_req > n_ctx:
            self.fail(
                "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n"
                "either reduce n_predict or increase n_ctx."
            )

        if not self.disable_log:
            # print the prompt token-by-token
            prompt = ""
            for i in tokens_list:
                prompt += cy.common_token_to_piece(self.ctx, i)
            self.log.info(prompt.strip())

        # create a llama_batch with size 512
        # we use this object to submit token data for decoding

        # create batch
        batch = cy.LlamaBatch(n_tokens=512, embd=0, n_seq_max=1)

        # evaluate the initial prompt
        for i, token in enumerate(tokens_list):
            cy.common_batch_add(batch, token, i, [0], False)

        # llama_decode will output logits only for the last token of the prompt
        # batch.logits[batch.n_tokens - 1] = True
        batch.set_last_logits_to_true()

        # logits = batch.get_logits()

        self.ctx.decode(batch)

        # main loop

        n_cur: int = batch.n_tokens
        n_decode: int = 0
        result: str = ""

        if not self.disable_log:
            t_main_start: int = cy.ggml_time_us()

        while n_cur <= n_predict:
            # sample the next token

            if True:
                new_token_id = self.smplr.sample(self.ctx, batch.n_tokens - 1)

                self.smplr.accept(new_token_id)

                # is it an end of generation?
                if self.model.token_is_eog(new_token_id) or n_cur == n_predict:
                    break

                result += cy.common_token_to_piece(self.ctx, new_token_id)

                # prepare the next batch
                cy.common_batch_clear(batch)
                # push this new token for next evaluation
                cy.common_batch_add(batch, new_token_id, n_cur, [0], True)

                n_decode += 1

            n_cur += 1

            # evaluate the current batch with the transformer model
            self.ctx.decode(batch)

        if not self.disable_log:
            t_main_end: int = cy.ggml_time_us()
            self.log.info(
                "decoded %d tokens in %.2f s, speed: %.2f t/s"
                % (
                    n_decode,
                    (t_main_end - t_main_start) / 1000000.0,
                    n_decode / ((t_main_end - t_main_start) / 1000000.0),
                )
            )

        return result.strip()
