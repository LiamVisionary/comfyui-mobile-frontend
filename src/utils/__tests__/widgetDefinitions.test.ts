import { describe, expect, it } from 'vitest';
import type { NodeTypes, Workflow, WorkflowNode } from '@/api/types';
import {
  getInputWidgetDefinitions,
  getWidgetDefinitions,
  PROXY_INDEX_OFFSET,
  resolveSubgraphPlaceholderInputWidgetDefs,
  resolveSubgraphProxyInputWidgetDefs,
} from '../widgetDefinitions';

function makeNode(id: number, type: string, widgetsValues: unknown[]): WorkflowNode {
  return {
    id,
    itemKey: `sk-${id}`,
    type,
    pos: [0, 0],
    size: [200, 100],
    flags: {},
    order: 0,
    mode: 0,
    inputs: [],
    outputs: [],
    properties: {},
    widgets_values: widgetsValues,
  };
}

describe('widgetDefinitions lora manager support', () => {
  it('does not skip an imaginary seed control slot for PromptAssistantGenerate', () => {
    const nodeTypes: NodeTypes = {
      PromptAssistantGenerate: {
        input: {
          required: {
            idea: ['STRING', { default: '' }],
            profile: [['swarm_booru_tags', 'swarm_wai_couple_two_lines'], { default: 'swarm_booru_tags' }],
            context: ['STRING', { default: '' }],
            image_caption: ['STRING', { default: '' }],
            extra_instructions: ['STRING', { default: '' }],
            timeout_seconds: ['FLOAT', { default: 60 }],
            seed: ['INT', { default: -1 }],
          },
          optional: {
            profile_json_override: ['STRING', { default: '' }],
            prompt: ['STRING', { default: '' }],
            negative_prompt: ['STRING', { default: '' }],
            helper_mode: [['None', 'Couple regions', 'Bounding boxes'], { default: 'None' }],
            emit_ui_text: ['BOOLEAN', { default: false }],
            auto_generate_on_queue: ['BOOLEAN', { default: false }],
          },
        },
        input_order: {
          required: ['idea', 'profile', 'context', 'image_caption', 'extra_instructions', 'timeout_seconds', 'seed'],
          optional: ['profile_json_override', 'prompt', 'negative_prompt', 'helper_mode', 'emit_ui_text', 'auto_generate_on_queue'],
        },
        output: [],
        output_name: [],
        name: 'PromptAssistantGenerate',
        display_name: 'Prompt Assistant Generate',
        description: '',
        python_module: '',
        category: '',
      },
    };
    const prompt = 'top-bottom composition\n1girl, himawari\n1boy, naruto';
    const negative = 'close-up, cropped';
    const node = makeNode(12, 'PromptAssistantGenerate', [
      'idea',
      'swarm_booru_tags',
      '',
      '',
      '',
      90,
      4321,
      '',
      prompt,
      negative,
      'Couple regions',
      true,
      false,
    ]);

    const widgets = getWidgetDefinitions(nodeTypes, node);
    const inputWidgets = getInputWidgetDefinitions(nodeTypes, node);

    expect(widgets.find((widget) => widget.name === 'prompt')).toMatchObject({
      widgetIndex: 8,
      value: prompt,
    });
    expect(widgets.find((widget) => widget.name === 'negative_prompt')).toMatchObject({
      widgetIndex: 9,
      value: negative,
    });
    expect(inputWidgets.find((widget) => widget.name === 'helper_mode')).toMatchObject({
      widgetIndex: 10,
      value: 'Couple regions',
    });
  });

  it('builds lora manager synthetic widgets with choices from LoraLoader', () => {
    const nodeTypes: NodeTypes = {
      LoraLoader: {
        input: {
          required: {
            lora_name: ['COMBO', { choices: ['a.safetensors', 'b.safetensors'] }],
          },
        },
        output: [],
        output_name: [],
        name: 'LoraLoader',
        display_name: 'LoraLoader',
        description: '',
        python_module: '',
        category: '',
      },
    };

    const node = makeNode(1, 'Lora Loader (LoraManager)', [
      'text',
      [{ name: 'a.safetensors', strength: 1, active: true }],
    ]);

    const defs = getWidgetDefinitions(nodeTypes, node);
    expect(defs.map((d) => d.type)).toContain('LM_LORA_HEADER');
    expect(defs.map((d) => d.type)).toContain('LM_LORA');
    expect(defs.map((d) => d.type)).toContain('LM_LORA_ADD');

    const loraDef = defs.find((d) => d.type === 'LM_LORA');
    expect(loraDef?.options).toMatchObject({ entryIndex: 0 });
  });

  it('builds editable widgets for MultiLoRAStackModelOnly JSON stack nodes', () => {
    const nodeTypes: NodeTypes = {
      LoraLoader: {
        input: {
          required: {
            lora_name: [['anima-turbo-lora-v0.2.safetensors'], {}],
          },
        },
        output: [],
        output_name: [],
        name: 'LoraLoader',
        display_name: 'LoraLoader',
        description: '',
        python_module: '',
        category: '',
      },
    };

    const node = makeNode(11, 'MultiLoRAStackModelOnly', [
      '[{"on":true,"lora":"anima-turbo-lora-v0.2.safetensors","strength":0.85}]',
    ]);
    node.title = 'LOAD LORAS HERE - Multi LoRA Stack';

    const defs = getWidgetDefinitions(nodeTypes, node);
    expect(defs.map((def) => def.type)).toEqual([
      'LM_LORA_HEADER',
      'LM_LORA',
      'LM_LORA_ADD',
    ]);
    expect(defs[1]).toMatchObject({
      name: 'anima-turbo-lora-v0.2.safetensors',
      widgetIndex: 0,
      options: {
        entryIndex: 0,
        preserveFileExtension: true,
      },
      value: {
        name: 'anima-turbo-lora-v0.2.safetensors',
        strength: 0.85,
        active: true,
      },
    });
  });

  it('renders MFlux LoRA loader filename and strength widgets from object info', () => {
    const nodeTypes: NodeTypes = {
      MfluxLorasLoader: {
        input: {
          required: {
            Lora1: [['None', 'characters/naruto.safetensors'], {}],
            scale1: ['FLOAT', { default: 1, min: 0, max: 1, step: 0.01 }],
            Lora2: [['None', 'styles/anime.safetensors'], {}],
            scale2: ['FLOAT', { default: 1, min: 0, max: 1, step: 0.01 }],
            Lora3: [['None'], {}],
            scale3: ['FLOAT', { default: 1, min: 0, max: 1, step: 0.01 }],
          },
          optional: {
            Loras: ['MfluxLorasPipeline', {}],
          },
        },
        input_order: {
          required: ['Lora1', 'scale1', 'Lora2', 'scale2', 'Lora3', 'scale3'],
          optional: ['Loras'],
        },
        output: ['MfluxLorasPipeline'],
        output_name: ['Loras'],
        name: 'MfluxLorasLoader',
        display_name: 'MFlux Loras Loader',
        description: '',
        python_module: '',
        category: 'MFlux/Pro',
      },
    };

    const node = makeNode(11, 'MfluxLorasLoader', [
      'characters/naruto.safetensors',
      0.6,
      'styles/anime.safetensors',
      0.35,
      'None',
      1,
    ]);
    node.inputs = [{ name: 'Loras', type: 'MfluxLorasPipeline', link: null }];

    expect(getInputWidgetDefinitions(nodeTypes, node).map((def) => def.name)).toEqual([
      'Lora1',
      'Lora2',
      'Lora3',
    ]);
    expect(getWidgetDefinitions(nodeTypes, node).map((def) => [def.name, def.value])).toEqual([
      ['scale1', 0.6],
      ['scale2', 0.35],
      ['scale3', 1],
    ]);
  });

  it('renders COMFY_DYNAMICCOMBO_V3 inputs as editable combo keys', () => {
    const nodeTypes: NodeTypes = {
      ColorTransfer: {
        input: {
          required: {
            method: [['reinhard_lab'], {}],
            source_stats: [
              'COMFY_DYNAMICCOMBO_V3',
              {
                options: [
                  { key: 'per_frame', inputs: { required: {} } },
                  { key: 'uniform', inputs: { required: {} } },
                  { key: 'target_frame', inputs: { required: {} } },
                ],
              },
            ],
            strength: ['FLOAT', { default: 1 }],
          },
          optional: {},
        },
        input_order: {
          required: ['method', 'source_stats', 'strength'],
          optional: [],
        },
        output: [],
        output_name: [],
        name: 'ColorTransfer',
        display_name: 'ColorTransfer',
        description: '',
        python_module: '',
        category: '',
      },
    };
    const node = makeNode(11, 'ColorTransfer', [
      'reinhard_lab',
      { source_stats: 'per_frame' },
      0.8,
    ]);

    const defs = getInputWidgetDefinitions(nodeTypes, node);
    const sourceStats = defs.find((def) => def.name === 'source_stats');

    expect(sourceStats).toMatchObject({
      type: 'COMBO',
      value: { source_stats: 'per_frame' },
      widgetIndex: 1,
      options: {
        options: ['per_frame', 'uniform', 'target_frame'],
        __dynamicComboWidgetName: 'source_stats',
      },
    });
  });

  it('uses LoRA Manager widget ids to skip metadata widgets', () => {
    const nodeTypes: NodeTypes = {
      'Lora Loader (LoraManager)': {
        input: {
          required: {
            text: ['AUTOCOMPLETE_TEXT_LORAS', {}],
          },
          optional: {},
        },
        input_order: {
          required: ['text'],
          optional: [],
        },
        output: [],
        output_name: [],
        name: 'Lora Loader (LoraManager)',
        display_name: 'Lora Loader (LoraManager)',
        description: '',
        python_module: '',
        category: '',
      },
    };

    const node = makeNode(1, 'Lora Loader (LoraManager)', [
      { version: 1, textWidgetName: 'text' },
      '<lora:a:1.00>',
      [{ name: 'a', strength: 1, active: true }],
    ]);
    node.properties = {
      __lm_widget_ids: ['__lm_autocomplete_meta_text', 'text', 'loras'],
    };

    const defs = getWidgetDefinitions(nodeTypes, node);
    const textDef = defs.find((def) => def.name === 'text');
    expect(textDef).toMatchObject({
      value: '<lora:a:1.00>',
      widgetIndex: 1,
    });
    expect(defs.find((def) => def.type === 'LM_LORA')).toMatchObject({
      widgetIndex: 2,
    });
  });

  it('does not synthesize a phantom lora list for LoRA Text Loader nodes without a list widget', () => {
    const nodeTypes: NodeTypes = {
      'LoRA Text Loader (LoraManager)': {
        input: {
          required: {
            lora_syntax: ['STRING'],
          },
          optional: {},
        },
        input_order: {
          required: ['lora_syntax'],
          optional: [],
        },
        output: [],
        output_name: [],
        name: 'LoRA Text Loader (LoraManager)',
        display_name: 'LoRA Text Loader (LoraManager)',
        description: '',
        python_module: '',
        category: '',
      },
    };

    const node = makeNode(11, 'LoRA Text Loader (LoraManager)', ['<lora:foo:0.8>']);
    const defs = getWidgetDefinitions(nodeTypes, node);

    expect(defs).toHaveLength(1);
    expect(defs[0]).toMatchObject({
      name: 'lora_syntax',
      value: '<lora:foo:0.8>',
      widgetIndex: 0,
    });
    expect(defs.some((def) => def.type === 'LM_LORA')).toBe(false);
    expect(defs.some((def) => def.type === 'LM_LORA_ADD')).toBe(false);
  });

  it('builds trigger-word synthetic widgets and carries allowStrengthAdjustment', () => {
    const nodeTypes: NodeTypes = {
      'TriggerWord Toggle (LoraManager)': {
        input: {
          required: {
            allow_strength_adjustment: ['BOOLEAN', {}],
          },
          optional: {},
        },
        input_order: {
          required: ['allow_strength_adjustment'],
          optional: [],
        },
        output: [],
        output_name: [],
        name: 'TriggerWord Toggle (LoraManager)',
        display_name: 'TriggerWord Toggle (LoraManager)',
        description: '',
        python_module: '',
        category: '',
      },
    };

    const node = makeNode(2, 'TriggerWord Toggle (LoraManager)', [
      true,
      [{ text: 'foo', active: true, strength: 0.4 }],
      'foo',
    ]);

    const defs = getWidgetDefinitions(nodeTypes, node);
    const tw = defs.find((d) => d.type === 'TW_WORD');
    expect(tw?.options).toMatchObject({ entryIndex: 0, allowStrengthAdjustment: true });
  });

  it('builds standard widget definitions for regular nodes', () => {
    const nodeTypes: NodeTypes = {
      TestNode: {
        input: {
          required: {
            steps: ['INT', {}],
          },
          optional: {},
        },
        output: [],
        output_name: [],
        name: 'TestNode',
        display_name: 'TestNode',
        description: '',
        python_module: '',
        category: '',
      },
    };

    const node = makeNode(3, 'TestNode', [20]);
    const defs = getWidgetDefinitions(nodeTypes, node);
    expect(defs).toHaveLength(1);
    expect(defs[0]).toMatchObject({ name: 'steps', type: 'INT', value: 20 });
  });

  it('synthesizes proxied EasySeed control_after_generate from the inner seed control slot', () => {
    const innerSeed = makeNode(915, 'easy seed', [123, 'randomize', null]);
    innerSeed.title = 'EasySeed';
    innerSeed.outputs = [{ name: 'seed', type: 'INT', links: [] }];

    const placeholder = makeNode(911, 'subgraph-a', []);
    placeholder.properties = {
      proxyWidgets: [
        ['915', 'seed'],
        ['915', 'control_after_generate'],
      ],
    };

    const workflow: Workflow = {
      last_node_id: 915,
      last_link_id: 0,
      nodes: [placeholder],
      links: [],
      groups: [],
      config: {},
      version: 1,
      definitions: {
        subgraphs: [
          {
            id: 'subgraph-a',
            nodes: [innerSeed],
            links: [],
            groups: [],
            config: {},
          },
        ],
      },
    };

    const inputDefs = resolveSubgraphProxyInputWidgetDefs(
      placeholder,
      workflow,
      null,
    );

    expect(inputDefs).toHaveLength(1);
    expect(inputDefs[0]).toMatchObject({
      name: 'EasySeed: control_after_generate',
      type: 'COMBO',
      value: 'randomize',
      widgetIndex: PROXY_INDEX_OFFSET + 1,
      options: {
        options: ['fixed', 'randomize', 'increment', 'decrement'],
        __proxy: {
          subgraphId: 'subgraph-a',
          innerNodeId: 915,
          innerWidgetIndex: 1,
        },
      },
    });
  });

  it('resolves promoted subgraph placeholder combo values from linked source nodes', () => {
    const sourceNode = makeNode(100, 'PrimitiveNode', ['euler']);
    sourceNode.outputs = [{ name: 'sampler_name', type: 'COMBO', links: [55] }];

    const placeholder = makeNode(200, 'subgraph-a', []);
    placeholder.inputs = [
      {
        name: 'sampler_name',
        type: 'COMBO',
        link: 55,
        widget: { name: 'sampler_name' },
      },
    ];

    const innerNode = makeNode(300, 'SamplerNode', []);
    const workflow: Workflow = {
      last_node_id: 300,
      last_link_id: 55,
      nodes: [sourceNode, placeholder],
      links: [[55, sourceNode.id, 0, placeholder.id, 0, 'COMBO']],
      groups: [],
      config: {},
      version: 1,
      definitions: {
        subgraphs: [
          {
            id: 'subgraph-a',
            nodes: [innerNode],
            links: [],
            groups: [],
            config: {},
          },
        ],
      },
    };
    const nodeTypes: NodeTypes = {
      SamplerNode: {
        input: {
          required: {
            sampler_name: [['euler', 'dpmpp_2m'], {}],
          },
          optional: {},
        },
        output: [],
        output_name: [],
        name: 'SamplerNode',
        display_name: 'SamplerNode',
        description: '',
        python_module: '',
        category: '',
      },
    };

    const inputDefs = resolveSubgraphPlaceholderInputWidgetDefs(
      placeholder,
      workflow,
      nodeTypes,
    );

    expect(inputDefs).toHaveLength(1);
    expect(inputDefs[0]).toMatchObject({
      name: 'sampler_name',
      type: 'COMBO',
      value: 'euler',
      widgetIndex: 0,
      options: {
        options: ['euler', 'dpmpp_2m'],
        __linkedSource: {
          subgraphId: null,
          nodeId: sourceNode.id,
          widgetIndex: 0,
          widgetName: 'sampler_name',
          itemKey: sourceNode.itemKey,
        },
      },
    });
  });

  it('carries the model picker kind on a renamed promoted model-loader widget', () => {
    const placeholder = makeNode(200, 'subgraph-ckpt', ['model.safetensors']);
    placeholder.inputs = [
      {
        name: 'ckpt_name',
        // Renamed promoted label — name-based detection would miss it.
        localized_name: 'Checkpoint',
        type: 'COMBO',
        link: null,
        widget: { name: 'ckpt_name' },
      },
    ];

    const innerNode = makeNode(300, 'CheckpointLoaderSimple', []);
    const workflow: Workflow = {
      last_node_id: 300,
      last_link_id: 0,
      nodes: [placeholder],
      links: [],
      groups: [],
      config: {},
      version: 1,
      definitions: {
        subgraphs: [
          {
            id: 'subgraph-ckpt',
            nodes: [innerNode],
            links: [],
            groups: [],
            config: {},
          },
        ],
      },
    };
    const nodeTypes: NodeTypes = {
      CheckpointLoaderSimple: {
        input: {
          required: { ckpt_name: [['model.safetensors', 'other.safetensors'], {}] },
          optional: {},
        },
        output: [],
        output_name: [],
        name: 'CheckpointLoaderSimple',
        display_name: 'Load Checkpoint',
        description: '',
        python_module: '',
        category: '',
      },
    };

    const inputDefs = resolveSubgraphPlaceholderInputWidgetDefs(
      placeholder,
      workflow,
      nodeTypes,
    );

    expect(inputDefs).toHaveLength(1);
    // Shown under its display label, but the picker kind is detected from the
    // inner ComfyUI input name (ckpt_name -> checkpoints).
    expect(inputDefs[0].name).toBe('Checkpoint');
    expect((inputDefs[0].options as Record<string, unknown>).__modelKind).toBe(
      'checkpoints',
    );
  });
});
