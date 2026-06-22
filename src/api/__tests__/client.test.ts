import { describe, expect, it } from 'vitest';
import { cloneWithCompatibleLoraChoices } from '../client';

describe('cloneWithCompatibleLoraChoices', () => {
  it('keeps ComfyUI server LoRA choices while adding wrapper-compatible LoRAs', () => {
    const nodeTypes = {
      LoraLoader: {
        input: {
          required: {
            lora_name: [
              ['FK_cumstrings.safetensors', 'ExistingServerLoRA.safetensors'],
              { default: 'ExistingServerLoRA.safetensors' },
            ],
          },
        },
      },
    };

    const cloned = cloneWithCompatibleLoraChoices(nodeTypes as never, [
      { id: 'ZImageTurboOnly.safetensors', name: 'ZImageTurboOnly.safetensors' },
    ]) as unknown as typeof nodeTypes;

    expect(cloned.LoraLoader.input.required.lora_name[0]).toEqual([
      'FK_cumstrings.safetensors',
      'ExistingServerLoRA.safetensors',
      'ZImageTurboOnly.safetensors',
    ]);
  });
});
