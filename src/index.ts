export { completion } from './completion';
export { embedding } from './embedding';
import { AnthropicHandler } from './handlers/anthropic';
import { CohereHandler } from './handlers/cohere';
import { OllamaHandler } from './handlers/ollama';
import { OpenAIHandler } from './handlers/openai';
import { AI21Handler } from './handlers/ai21';
import { ReplicateHandler } from './handlers/replicate';
import { DeepInfraHandler } from './handlers/deepinfra';
import { MistralHandler } from './handlers/mistral';
import {
  Handler,
  HandlerParams,
  HandlerParamsNotStreaming,
  HandlerParamsStreaming,
  Result,
  ResultNotStreaming,
  ResultStreaming,
} from './types';

interface ProviderParams {
  apiKey: string;
  baseUrl: string;
  providerType: ProviderType;
}

enum ProviderType {
  OpenAI,
  Mistral,
}

export function getHandler(
  providerType: ProviderType,
  mapping: Record<ProviderType, Handler>,
): Handler | null {
  return mapping[providerType] || null;
}

class Provider {
  private apiKey: string;
  private baseUrl: string;
  private providerType: ProviderType;

  constructor(params: ProviderParams) {
    this.apiKey = params.apiKey;
    this.baseUrl = params.baseUrl;
    this.providerType = params.providerType;
  }

  PROVIDER_TYPE_HANDLER_MAPPINGS: Record<ProviderType, Handler> = {
    [ProviderType.OpenAI]: OpenAIHandler,
    [ProviderType.Mistral]: MistralHandler,
  };

  async completion(
    params: HandlerParamsNotStreaming,
  ): Promise<ResultNotStreaming>;

  async completion(params: HandlerParamsStreaming): Promise<ResultStreaming>;

  async completion(params: HandlerParams): Promise<Result>;

  async completion(params: HandlerParams): Promise<Result> {
    const handler = getHandler(
      this.providerType,
      this.PROVIDER_TYPE_HANDLER_MAPPINGS,
    );

    if (!handler) {
      throw new Error(
        `Model: ${params.model} not supported. Cannot find a handler.`,
      );
    }

    return handler(params);
  }
}
