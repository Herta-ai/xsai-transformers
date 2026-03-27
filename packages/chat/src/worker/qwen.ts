import type { PipelineOptionsFrom } from '@xsai-transformers/shared/types'
import type { GenerateTextResponse } from '@xsai/generate-text'
import type { AssistantMessage } from '@xsai/shared-chat'
import type { PreTrainedModel, Processor, Tensor } from 'transformers-next'

import { defineInvokeHandler, defineStreamInvokeHandler, nanoid, toStreamHandler } from '@moeru/eventa'
import { createContext } from '@moeru/eventa/adapters/webworkers/worker'
import { merge } from '@moeru/std/merge'
import { isWebGPUSupported } from 'gpuu/webgpu'
import { AutoProcessor, Qwen3_5ForConditionalGeneration, TextStreamer } from 'transformers-next'

import { chatCompletion, load } from '../shared'
import { MessageStatus } from '../types'

const { context } = createContext()

// eslint-disable-next-line @masknet/no-top-level
let chat: PreTrainedModel
// eslint-disable-next-line @masknet/no-top-level
let chatModelId: string
// eslint-disable-next-line @masknet/no-top-level
let processor: Processor

function generateTextResponseFromChoices(choices: GenerateTextResponse['choices'], usage: {
  completion_tokens: number
  prompt_tokens: number
  total_tokens: number
}): GenerateTextResponse {
  return {
    choices,
    created: Math.floor(Date.now() / 1000),
    id: nanoid(),
    model: chatModelId,
    object: 'chat.completion',
    system_fingerprint: '',
    usage,
  }
}

// eslint-disable-next-line @masknet/no-top-level
defineInvokeHandler(context, chatCompletion, async ({ messages, options }) => {
  const text = processor.apply_chat_template(messages, {
    add_generation_prompt: true,
  })
  const inputs = await processor(text)
  const outputs = await chat.generate(({
    ...inputs,
    max_new_tokens: 1024,
    streamer: new TextStreamer(processor.tokenizer!, {
      skip_prompt: true,
      skip_special_tokens: true,
    }),
  })) as Tensor
  const result = processor.batch_decode(
    outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
    {
      skip_special_tokens: true,
    },
  )
  return generateTextResponseFromChoices([
    {
      finish_reason: 'stop',
      index: 0,
      message: { content: result[0], role: 'assistant' as AssistantMessage['role'] },
    },
  ], {
    completion_tokens: 0,
    prompt_tokens: 0,
    total_tokens: 0,
  })
})

// eslint-disable-next-line @masknet/no-top-level
defineStreamInvokeHandler(context, load, toStreamHandler(async ({ emit, payload: { modelId, options } }) => {
  const device = (await isWebGPUSupported()) ? 'webgpu' : 'wasm'

  const opts = merge<PipelineOptionsFrom<typeof pipeline<'text-generation'>>>({
    device,
    progress_callback: (p) => {
      emit({ data: { progress: p }, type: 'progress' })
    },
  }, options)

  emit({ data: { message: `Using device: "${device}"` }, type: 'info' })
  emit({ data: { message: 'Loading models...' }, type: 'info' })

  processor = await AutoProcessor.from_pretrained(modelId)
  chat = await Qwen3_5ForConditionalGeneration.from_pretrained(modelId, opts)

  chatModelId = modelId

  emit({ data: { message: 'Ready!', status: MessageStatus.Ready }, type: 'status' })
}))
