# Quero-Quero

Continual pretraining PT-BR do `Polygl0t/Tucano2-0.6B-Base` com dados reais e
exportação de um artefato Hugging Face verificável.

## Estado

O projeto ainda contém apenas inventário, decisões e um protótipo inativo. Não
há pipeline executável, configuração de treino ou testes.

## Modelo

```yaml
model_id: Polygl0t/Tucano2-0.6B-Base
revision: dad97dc864a8f9a1d240fb9351d098f3af9511d7
native_context_length: 4096
training_sequence_length: 1024
training_method: full_parameter_continual_pretraining
```

O tokenizer, vocabulário, IDs e special tokens não podem mudar. Pesos refinados
permanecem `internal_research_only` até a revisão das permissões dos corpora.

## Datasets

| Dataset | Fonte | Estado |
| --- | --- | --- |
| [BrWaC-CLEAN](docs/datasets/brwac.md) | `BrWac.zip` → `data/*.txt` | selecionado |
| [WackyWacky](docs/datasets/wackywacky.md) | `pages.tsv` filtrado | selecionado |
| [MultiWOZ-PTBR](docs/datasets/multiwoz-ptbr.md) | 17 arquivos `dialogues_*.json` | selecionado |
| [OuterSpace](docs/datasets/outerspace.md) | `conversations.zip` | selecionado |
| [Adrenaline](docs/datasets/adrenaline.md) | `conversations.zip` | selecionado |
| [GigaVerbo-v2](docs/datasets/gigaverbo.md) | `Polygl0t/gigaverbo-v2` | desabilitado |

Todos os limites por documentos ou tokens devem ser configuráveis.

## Acesso aos dados

Os datasets ficam fora do Git e são fornecidos por uma raiz absoluta somente
leitura:

```sh
export PTBR_DATASET_ROOT=/caminho/absoluto/para/os/datasets
```

No ambiente local, `./mount_remote.sh` monta a origem em `.remote-datasets/`;
ambos permanecem ignorados. Derivados são gravados apenas em diretórios locais
também ignorados.

## Pipeline esperado

1. validar caminhos e fingerprints das fontes;
2. aplicar seleção, limpeza e tokenização configuráveis;
3. gerar shards pequenos, retomáveis e com proveniência;
4. treinar com seeds e configuração versionadas;
5. avaliar qualidade PT-BR;
6. exportar o modelo conforme o
   [contrato do artefato](docs/model-artifact-contract.md).

Dados, manifests detalhados, shards, caches, checkpoints e pesos nunca entram no
Git. Métricas e manifests ficam separados dos checkpoints.

## Implementação atual

`requirements.txt` não é lockfile. O
`prototypes/gigaverbo_streaming.py` é apenas exploratório e não deve ser
executado enquanto o GigaVerbo estiver desabilitado.
