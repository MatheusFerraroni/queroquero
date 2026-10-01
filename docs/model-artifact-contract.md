# Contrato do artefato do modelo

Versão: `tucano2-model-artifact/v1`.

O produtor e o consumidor compartilham somente este contrato e o diretório
exportado.

## Baseline

```yaml
kind: huggingface
model_id: Polygl0t/Tucano2-0.6B-Base
revision: dad97dc864a8f9a1d240fb9351d098f3af9511d7
sequence_length: 1024
result_variant: upstream_baseline
```

Revisões móveis e checkpoints intermediários não são aceitos.

## Compatibilidade

O artefato refinado deve:

- manter arquitetura, tokenizer, vocabulário, IDs e special tokens do baseline;
- preservar contexto nativo de 4.096 e registrar treino com 1.024 tokens;
- conter pesos completos em `safetensors`, sem adapters isolados;
- carregar offline com `AutoModelForCausalLM` e `AutoTokenizer`;
- permanecer fora do Git e sem symlinks.

Mudanças de arquitetura ou tokenizer exigem nova versão do contrato.

## Layout mínimo

```text
<artifact>/
├── config.json
├── model.safetensors
│   ou shards + model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json        # inclui os special tokens no Transformers 5
├── special_tokens_map.json      # opcional, para layouts legados
├── generation_config.json        # quando gerado
└── model_artifact_manifest.json
```

Arquivos adicionais são permitidos se declarados no manifesto. Datasets,
métricas detalhadas e checkpoints de retomada não pertencem ao artefato.

## Manifesto

`model_artifact_manifest.json` usa UTF-8, chaves determinísticas e contém:

| Campo | Conteúdo obrigatório |
| --- | --- |
| `schema_version` | `tucano2-model-artifact/v1` |
| `artifact_id` | identificador estável |
| `format` | `transformers_pretrained` |
| `parent_model` | model ID, revisão e licença |
| `architecture` | model type, parâmetros, contextos 4.096/1.024 |
| `tokenizer` | fingerprint, vocab size e IDs especiais |
| `training` | método, commit, run ID, seed, passos positivos, estratégia, world size, batch, precisão, otimizador e hashes de config/dataset; no perfil real pareado, braço, plano, schedule, pools preparados/consumidos e digest comum dos inputs |
| `environment` | versões de Python, torch, transformers e tokenizers |
| `files` | path relativo, bytes e SHA-256 de cada arquivo |
| `artifact_sha256` | hash agregado |
| `redistribution_status` | `internal_research_only` |

Regras de hash:

- `files` exclui o manifesto e é ordenado lexicograficamente;
- cada SHA-256 cobre os bytes exatos do arquivo;
- o hash agregado cobre a lista ordenada de `path`, `size_bytes` e `sha256`;
- o fingerprint do tokenizer cobre seus arquivos e IDs especiais;
- `dataset_manifest_sha256` referencia um manifesto externo.

O manifesto não pode conter caminhos absolutos, hosts, usernames, textos,
segredos ou valores pessoais.

## Carregamento pelo consumidor

```yaml
# baseline
model:
  kind: huggingface
  model_id: Polygl0t/Tucano2-0.6B-Base
  revision: dad97dc864a8f9a1d240fb9351d098f3af9511d7
  sequence_length: 1024

# refinado
model:
  kind: local_artifact
  expected_schema: tucano2-model-artifact/v1
  expected_artifact_sha256: <sha256>
  sequence_length: 1024
```

O caminho local é absoluto e fornecido em execução. O consumidor rejeita
manifesto inválido, hashes divergentes, arquivos ausentes/extras, symlinks,
tokenizer incompatível, contexto menor que 1.024, adapter isolado ou revisão
móvel.

## Resultados e distribuição

Baseline e modelo refinado usam variantes distintas. Ao trocar o modelo, os
experimentos reiniciam sem reutilizar pesos, optimizer ou checkpoints.

A licença Apache-2.0 do modelo pai não autoriza redistribuição dos pesos
refinados. O status permanece `internal_research_only` até revisão explícita das
fontes.
