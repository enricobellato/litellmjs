import { IProviderWrapper } from '../types';
import {
  HandlerParams,
  ResultStreaming,
  ResultNotStreaming,
  Result,
  StreamingChunk,
} from '../types';
import { getUnixTimestamp } from '../utils/getUnixTimestamp';
import { combinePrompts } from '../utils/combinePrompts';
import { toUsage } from '../utils/toUsage';

interface OllamaResponseChunk {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
}

class OllamaWrapper implements IProviderWrapper {
  private baseUrl: string;

  constructor(apiKey?: string, baseUrl?: string) {
    this.baseUrl = baseUrl ?? 'http://127.0.0.1:11434';
  }

  private toStreamingChunk(
    ollamaResponse: OllamaResponseChunk,
    model: string,
    prompt: string,
  ): StreamingChunk {
    return {
      model: model,
      created: getUnixTimestamp(),
      usage: toUsage(prompt, ollamaResponse.response),
      choices: [
        {
          delta: { content: ollamaResponse.response, role: 'assistant' },
          finish_reason: 'stop',
          index: 0,
        },
      ],
    };
  }

  private toResponse(
    content: string,
    model: string,
    prompt: string,
  ): ResultNotStreaming {
    return {
      model: model,
      created: getUnixTimestamp(),
      usage: toUsage(prompt, content),
      choices: [
        {
          message: { content, role: 'assistant' },
          finish_reason: 'stop',
          index: 0,
        },
      ],
    };
  }

  private async *iterateResponse(
    response: Response,
    model: string,
    prompt: string,
  ): AsyncIterable<StreamingChunk> {
    const reader = response.body?.getReader();
    let done = false;

    while (!done) {
      const next = await reader?.read();
      if (next?.value) {
        const decoded = new TextDecoder().decode(next.value);
        done = next.done;
        const lines = decoded.split(/(?<!\\)\n/);
        const ollamaResponses = lines
          .map((line) => line.trim())
          .filter((line) => line !== '')
          .map((line) => JSON.parse(line) as OllamaResponseChunk)
          .map((response) => this.toStreamingChunk(response, model, prompt));

        yield* ollamaResponses;
      } else {
        done = true;
      }
    }
  }

  private async getOllamaResponse(
    model: string,
    prompt: string,
    baseUrl: string,
  ): Promise<Response> {
    return fetch(`${baseUrl}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        prompt,
      }),
    });
  }

  public async completions(
    params: HandlerParams & { stream: true },
  ): Promise<ResultStreaming>;
  public async completions(
    params: HandlerParams & { stream?: false },
  ): Promise<ResultNotStreaming>;
  public async completions(
    params: HandlerParams & { stream?: boolean },
  ): Promise<Result> {
    const model = params.model;
    const prompt = combinePrompts(params.messages);

    const res = await this.getOllamaResponse(model, prompt, this.baseUrl);

    if (!res.ok) {
      throw new Error(
        `Received an error with code ${res.status} from Ollama API.`,
      );
    }

    if (params.stream) {
      return this.iterateResponse(res, model, prompt);
    }

    const chunks: StreamingChunk[] = [];

    for await (const chunk of this.iterateResponse(res, model, prompt)) {
      chunks.push(chunk);
    }

    const message = chunks.reduce((acc: string, chunk: StreamingChunk) => {
      return (acc += chunk.choices[0].delta.content);
    }, '');

    return this.toResponse(message, model, prompt);
  }
}

export default OllamaWrapper;
