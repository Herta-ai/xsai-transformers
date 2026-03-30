import type { pipeline } from '@huggingface/transformers'
import type { PipelineOptionsFrom } from '@xsai-transformers/shared/types'

import type {ConfigureParams, EmbedWorkerParams, EmbedWorkerResults, LoadParams} from './types'

import { createLoadDefinition, createProcessDefinition } from '../../shared/src/worker/rpc'

export const load = createLoadDefinition<LoadParams<Omit<PipelineOptionsFrom<typeof pipeline<'feature-extraction'>>, 'progress_callback'>>>()
export const extract = createProcessDefinition<EmbedWorkerParams, EmbedWorkerResults>('extract')
export const configure = createProcessDefinition<ConfigureParams, void>('configure')
