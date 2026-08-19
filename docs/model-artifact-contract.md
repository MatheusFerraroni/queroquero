# Contrato de artefato do modelo

Versão do contrato: `tucano2-model-artifact/v1`.

Este documento define a única interface entre o projeto que refina o Tucano 2
0.6B e o projeto que executa os experimentos federados. Os projetos não
compartilham código, dependências, datasets, diretórios de execução nem caminhos
relativos.

## Baseline padrão

Enquanto o modelo refinado não estiver disponível, o consumidor usa:

```yaml
kind: huggingface
model_id: Polygl0t/Tucano2-0.6B-Base
revision: dad97dc864a8f9a1d240fb9351d098f3af9511d7
sequence_length: 1024
result_variant: upstream_baseline
```

A revisão é imutável e identifica o checkpoint final publicado. Não é permitido
substituí-la silenciosamente por `main`, por uma tag móvel ou pelo checkpoint
intermediário `step-160000-end-of-stage-2`.

## Invariantes de compatibilidade

O artefato refinado deve:

- usar a mesma arquitetura do Tucano 2 0.6B padrão;
- manter tokenizer, vocabulário, IDs e special tokens sem alterações;
- preservar o contexto arquitetural nativo de 4.096 tokens;
- registrar que preparação e experimentos usam sequências de 1.024 tokens;
- conter todos os pesos do modelo, não apenas deltas, adapters ou um checkpoint
  interno do treinador;
- ser carregável com `AutoModelForCausalLM.from_pretrained()` e
  `AutoTokenizer.from_pretrained()` apontando somente para o diretório local;
- usar arquivos `safetensors` para os pesos;
- permanecer fora do Git.

No contrato v1, uma mudança de tokenizer, vocabulário, arquitetura ou special
tokens torna o artefato incompatível e exige uma nova versão do contrato.

## Layout do artefato refinado

```text
<model-artifact-dir>/
├── config.json
├── model.safetensors
│   ou model-00001-of-*.safetensors + model.safetensors.index.json
├── tokenizer.json ou os arquivos equivalentes do tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── generation_config.json             # quando produzido pela biblioteca
└── model_artifact_manifest.json
```

Arquivos adicionais gerados pelo formato Hugging Face são permitidos desde que
sejam declarados no manifesto. Symlinks, datasets, métricas detalhadas,
checkpoints de retomada e caminhos pessoais não pertencem ao artefato.

## Manifesto obrigatório

`model_artifact_manifest.json` usa UTF-8, chaves em ordem determinística e
contém, no mínimo:

```json
{
  "schema_version": "tucano2-model-artifact/v1",
  "artifact_id": "<identificador-estavel>",
  "format": "transformers_pretrained",
  "parent_model": {
    "model_id": "Polygl0t/Tucano2-0.6B-Base",
    "revision": "dad97dc864a8f9a1d240fb9351d098f3af9511d7",
    "license": "Apache-2.0"
  },
  "architecture": {
    "model_type": "<tipo-em-config.json>",
    "parameter_count": "<inteiro>",
    "native_context_length": 4096,
    "training_sequence_length": 1024
  },
  "tokenizer": {
    "fingerprint_sha256": "<sha256>",
    "vocab_size": "<inteiro>",
    "bos_token_id": "<inteiro-ou-null>",
    "eos_token_id": "<inteiro-ou-null>",
    "pad_token_id": "<inteiro-ou-null>",
    "unk_token_id": "<inteiro-ou-null>"
  },
  "training": {
    "method": "full_parameter_continual_pretraining",
    "producer_git_commit": "<sha>",
    "run_id": "<id>",
    "seed": "<inteiro>",
    "resolved_config_sha256": "<sha256>",
    "dataset_manifest_sha256": "<sha256>"
  },
  "environment": {
    "python": "<versao>",
    "torch": "<versao>",
    "transformers": "<versao>",
    "tokenizers": "<versao>"
  },
  "files": [
    {
      "path": "<caminho-relativo>",
      "size_bytes": "<inteiro>",
      "sha256": "<sha256>"
    }
  ],
  "artifact_sha256": "<sha256-agregado>",
  "redistribution_status": "internal_research_only"
}
```

O manifesto não pode conter caminhos absolutos, hosts, usernames, URLs de
registros dos corpora, textos-fonte nem valores pessoais.

## Fingerprints

- `files` lista todos os arquivos regulares do artefato, exceto o próprio
  manifesto, em ordem lexicográfica pelo caminho relativo.
- Cada hash usa SHA-256 sobre os bytes exatos do arquivo.
- `artifact_sha256` usa uma serialização determinística da lista ordenada de
  `path`, `size_bytes` e `sha256`.
- O fingerprint do tokenizer cobre todos os seus arquivos declarados e os IDs
  especiais resolvidos.
- O hash do dataset manifest referencia o manifesto externo do produtor; o
  dataset manifest e os dados reais não são copiados para o artefato.

## Seleção e carregamento pelo consumidor

O consumidor aceita dois modos documentais:

```yaml
model:
  kind: huggingface
  model_id: Polygl0t/Tucano2-0.6B-Base
  revision: dad97dc864a8f9a1d240fb9351d098f3af9511d7
  sequence_length: 1024
```

```yaml
model:
  kind: local_artifact
  expected_schema: tucano2-model-artifact/v1
  expected_artifact_sha256: <sha256>
  sequence_length: 1024
```

No modo local, o diretório é fornecido futuramente por argumento absoluto de
execução. O caminho não é versionado e não pode apontar por `..` para outro
projeto.

Antes de treinar, o consumidor deverá rejeitar:

- manifesto ausente, incompleto ou de versão desconhecida;
- hash agregado ou hash de arquivo divergente;
- arquivos obrigatórios ausentes, extras não declarados ou symlinks;
- arquitetura, vocabulário, tokenizer ou special tokens incompatíveis;
- contexto menor que 1.024;
- adapter isolado ou checkpoint que não seja um diretório Hugging Face completo;
- revisão móvel ou não pinada no modo Hugging Face.

## Isolamento dos resultados

Resultados produzidos com o modelo padrão recebem a variante
`upstream_baseline`. Resultados com o modelo refinado recebem uma variante
distinta vinculada ao `artifact_sha256`.

Quando o artefato refinado chegar:

- todos os cenários B0 e F0-F5 começam novamente da rodada zero;
- pesos, estados de optimizer, checkpoints e updates federados do baseline não
  são reutilizados;
- a mesma especificação sintética e as mesmas seeds podem ser regeneradas para
  comparação pareada;
- resultados do baseline não podem ser apresentados como resultados finais do
  modelo refinado.

## Distribuição e evolução

A licença Apache-2.0 do modelo pai não determina a permissão de redistribuir
pesos refinados com os corpora selecionados. O valor padrão permanece
`internal_research_only` até uma revisão explícita de licenças, termos,
autorizações e privacidade.

Qualquer mudança incompatível cria uma nova versão do schema. Enquanto as duas
pastas estiverem no mesmo repositório, as duas cópias deste documento devem ser
byte a byte idênticas. Depois da separação, uma mudança exige o mesmo número de
versão e a mesma semântica nos dois repositórios.
