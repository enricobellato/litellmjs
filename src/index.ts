import OpenAIWrapper from './providers/openai';
import OllamaWrapper from './providers/ollama';
import {
  Handler,
  HandlerParams,
  HandlerParamsNotStreaming,
  HandlerParamsStreaming,
  Result,
  ResultNotStreaming,
  ResultStreaming,
  IProviderWrapper,
} from './types';

interface ProviderParams {
  apiKey: string;
  baseUrl: string;
  providerType: ProviderType;
}

enum ProviderType {
  OpenAI,
  Ollama,
}

export function getHandler(
  providerType: ProviderType,
  mapping: Record<ProviderType, Handler>,
): Handler | null {
  return mapping[providerType] || null;
}

export class Provider {
  private apiKey: string;
  private baseUrl: string;
  private providerType: ProviderType;
  private static PROVIDER_TYPE_HANDLER_MAPPINGS: Record<
    ProviderType,
    (apiKey: string, baseUrl: string) => IProviderWrapper
  > = {
      [ProviderType.OpenAI]: (apiKey, baseUrl) =>
      new OpenAIWrapper(apiKey, baseUrl),
      [ProviderType.Ollama]: (apiKey, baseUrl) =>
      new OllamaWrapper(apiKey, baseUrl),
  };

  constructor(params: ProviderParams) {
    this.apiKey = params.apiKey;
    this.baseUrl = params.baseUrl;
    this.providerType = params.providerType;
  }

  async completion(
    params: HandlerParamsNotStreaming & { stream: false },
  ): Promise<ResultNotStreaming>;

  async completion(
    params: HandlerParamsStreaming & { stream?: true },
  ): Promise<ResultStreaming>;

  async completion(params: HandlerParams): Promise<Result> {
    const clientCreationFunction =
      Provider.PROVIDER_TYPE_HANDLER_MAPPINGS[this.providerType];

    // Handle the case where there is no mapping for the given providerType
    if (!clientCreationFunction) {
      throw new Error(
        `Provider not supported for provider type: ${this.providerType}`,
      );
    }

    // Instantiate the correct provider wrapper
    const client = clientCreationFunction(this.apiKey, this.baseUrl);

    // Call the completions method on the handler with necessary params
    if (params.stream === true) {
      return client.completions(
        params as HandlerParamsStreaming & { stream: true },
      );
    } else {
      return client.completions(params as HandlerParamsNotStreaming);
    }
  }
}
