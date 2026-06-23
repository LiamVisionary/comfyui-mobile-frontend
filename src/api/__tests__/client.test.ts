import { afterEach, describe, expect, it, vi } from 'vitest';
import { detectNativeMlxBigLoveKlein3, queuePrompt, searchUserImagesByPrompt } from '@/api/client';

describe('searchUserImagesByPrompt', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('unions name/path and prompt searches without trusting directory entries', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const params = new URL(`http://localhost${url}`).searchParams;
      const files = params.has('search')
        ? [
            { name: 'video', path: 'video', type: 'dir', date: 1 },
            {
              name: 'ComfyUI_04555_.png',
              path: '.hidden/batch/sample scene/ComfyUI_04555_.png',
              folder: '.hidden/batch/sample scene',
              type: 'image',
              date: 2,
              size: 100,
            },
          ]
        : [
            {
              name: 'ComfyUI_04555_.png',
              path: '.hidden/batch/sample scene/ComfyUI_04555_.png',
              folder: '.hidden/batch/sample scene',
              type: 'image',
              date: 2,
              size: 100,
            },
            {
              name: 'ComfyUI_04556_.png',
              path: '.hidden/batch/sample scene/ComfyUI_04556_.png',
              folder: '.hidden/batch/sample scene',
              type: 'image',
              date: 3,
              size: 101,
            },
          ];

      return {
        ok: true,
        json: async () => ({ files, total: files.length, offset: 0, limit: 0 }),
      } as Response;
    });

    vi.stubGlobal('fetch', fetchMock);

    const results = await searchUserImagesByPrompt('output', 'sample scene', null, true);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls.some((url) => url.includes('search=sample+scene'))).toBe(true);
    expect(urls.some((url) => url.includes('prompt=sample+scene'))).toBe(true);
    expect(urls.some((url) => url.includes('q=sample+scene'))).toBe(false);
    expect(results.map((item) => item.id)).toEqual([
      'output/.hidden/batch/sample scene/ComfyUI_04555_.png',
      'output/.hidden/batch/sample scene/ComfyUI_04556_.png',
    ]);
  });
});

describe('detectNativeMlxBigLoveKlein3', () => {
  const bigLovePrompt = {
    '1': { class_type: 'UNETLoader', inputs: { unet_name: 'BigLoveKlein3_mxfp8.safetensors' } },
    '2': { class_type: 'LoadImage', inputs: { image: 'old-stale-image.png' } },
    '3': { class_type: 'LoadImage', inputs: { image: 'Screenshot 2026-06-21 at 8.52.09 PM.png' } },
    '4': { class_type: 'VAEEncode', inputs: { pixels: ['3', 0] } },
    '5': { class_type: 'KSampler', inputs: { latent_image: ['4', 0], positive: ['6', 0], steps: 4, seed: 123, cfg: 1 } },
    '6': { class_type: 'CLIPTextEncode', inputs: { text: 'add a red santa hat' } },
    '7': { class_type: 'EmptyLatentImage', inputs: { width: 768, height: 512 } },
  };

  it('uses the LoadImage wired into the sampler, not the first stale LoadImage', () => {
    expect(detectNativeMlxBigLoveKlein3(bigLovePrompt)?.imagePath).toBe('Screenshot 2026-06-21 at 8.52.09 PM.png');
  });

  it('keeps BigLove queueing on the normal prompt endpoint so the wrapper owns native routing', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ prompt_id: 'native-job-1', number: 0 }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }));
    vi.stubGlobal('fetch', fetchMock);

    await queuePrompt({ prompt: bigLovePrompt });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const call = fetchMock.mock.calls[0] as unknown as [RequestInfo | URL, RequestInit | undefined];
    const [url, init] = call;
    expect(String(url)).toBe('/api/prompt');
    const payload = JSON.parse(String(init?.body));
    expect(payload.prompt).toStrictEqual(bigLovePrompt);
  });
});
