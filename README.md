# Quero-Quero

Projeto de refinamento PT-BR do Tucano 2 0.6B.

Este projeto prepara dados reais em português brasileiro, executa o futuro
continual pretraining do Tucano 2 0.6B e exporta um artefato de modelo que possa
ser consumido por experimentos posteriores. Ele é autônomo: não depende de
código, arquivos ou caminhos relativos de outro projeto.

Esta primeira etapa contém somente documentação, o inventário de datasets e um
protótipo exploratório. Ainda não há pipeline de preparação ou treinamento
executável.

## Fronteira do projeto

Entram neste projeto:

- datasets reais em PT-BR autorizados para preparação e avaliação;
- inventário, amostragem e seleção explícita de fontes;
- limpeza, normalização, deduplicação e sanitização futuras;
- shards derivados, manifests e proveniência reproduzível;
- continual pretraining e avaliação de qualidade linguística;
- exportação verificável do modelo refinado.

Não entram neste projeto:

- geração de conversas ou segredos sintéticos;
- clientes, agregação ou treinamento federado;
- ataques de amplificação estrutural;
- DP-SGD, substituição semântica ou avaliação de vazamento;
- datasets, pesos, caches, checkpoints ou outros artefatos grandes no Git.

Dados reais podem ser usados somente no refinamento linguístico e na avaliação
externa de utilidade. Eles nunca devem ser usados como alvos de vazamento.

## Modelo e treinamento pretendido

O modelo pai está fixado em:

```yaml
model:
  model_id: Polygl0t/Tucano2-0.6B-Base
  revision: dad97dc864a8f9a1d240fb9351d098f3af9511d7
  native_context_length: 4096
  sequence_length: 1024
  training_method: full_parameter_continual_pretraining
```

O futuro treinamento atualizará todos os parâmetros do modelo. O tokenizer
original é parte imutável da interface: vocabulário, IDs e special tokens não
podem ser alterados. Embora o contexto nativo seja de 4.096 tokens, as
sequências de treino serão limitadas a 1.024 tokens.

O modelo pai usa licença Apache-2.0. Isso não resolve, por si só, as permissões
dos corpora usados no refinamento. Por isso, todo peso refinado deve ser marcado
como `internal_research_only` até a revisão das licenças, dos termos e das
permissões de cada fonte.

## Fontes selecionadas

As decisões atuais de artefatos são:

| Dataset | Fonte selecionada |
| --- | --- |
| BrWaC-CLEAN | `BrWac.zip` -> `data/*.txt`; `names.tsv` não entra como texto de treino |
| WackyWacky PT-BR | `pages.tsv`, usando linhas elegíveis após filtragem |
| MultiWOZ-PTBR | todos os 17 arquivos `dialogues_001.json` a `dialogues_017.json` |
| OuterSpace | seu próprio `conversations.zip` |
| Adrenaline | seu próprio `conversations.zip` |
| GigaVerbo-v2 | candidato remoto documentado; desabilitado por enquanto |

Adrenaline e OuterSpace são datasets distintos e independentes. Cada um terá
configuração, orçamento, manifesto, métricas e proveniência próprios.
`conversations_min.zip` serve apenas como fixture de desenvolvimento para seus
respectivos parsers. Os exports `forum.*.zip` não foram selecionados como fonte
de treino.

O inventário, as evidências estruturais e os filtros ainda pendentes estão em
[docs/datasets/](docs/datasets/README.md). Quantidades de documentos e tokens e
os filtros finais de qualidade ainda não foram definidos e não devem ser
inferidos silenciosamente.

As condições específicas de pré-ativação do GigaVerbo-v2 estão registradas em
[docs/datasets/gigaverbo.md](docs/datasets/gigaverbo.md); essa documentação não
altera seu estado desabilitado.

## Acesso externo aos datasets

Os datasets não pertencem a esta árvore. Todos os comandos documentados usam
`PTBR_DATASET_ROOT` como a raiz absoluta de um diretório externo e somente
leitura:

```sh
export PTBR_DATASET_ROOT=/caminho/absoluto/para/os/datasets
```

O valor real depende do ambiente e não deve ser salvo no Git. Código futuro
deverá rejeitar caminhos relativos e nunca escrever nessa raiz; manifests,
shards e outros derivados serão gravados em diretórios locais ignorados.

GigaVerbo permanece desabilitado nesta etapa. Se uma decisão futura o reativar,
ele deverá ser consumido por streaming, filtrado com `edu_int_score >= 4` e
limitado por uma quantidade configurável, sem download do corpus completo.

## Responsabilidades futuras

O pipeline deste projeto deverá:

1. identificar cada arquivo-fonte por fingerprint e registrar sua proveniência;
2. preparar subconjuntos configuráveis por documentos ou tokens;
3. criar shards derivados pequenos e retomáveis, sem reler os corpora completos
   em cada execução;
4. registrar configurações, seeds, versões, métricas e hashes separadamente dos
   checkpoints;
5. treinar e avaliar a qualidade PT-BR a partir da revisão pinada;
6. exportar um diretório Hugging Face completo, incluindo
   `model_artifact_manifest.json`, conforme o
   [contrato do artefato](docs/model-artifact-contract.md).

O artefato exportado deve permanecer fora do Git. O consumidor selecionará o
modelo por configuração e por um caminho absoluto externo ao repositório.

## Estado atual

- `docs/datasets/` contém somente inventários, estatísticas agregadas e decisões;
- `requirements.txt` preserva as dependências do protótipo inicial e ainda não é
  um lockfile do pipeline futuro;
- `prototypes/gigaverbo_streaming.py` foi preservado sem alterações apenas como
  experimento exploratório inativo; ele não é um entrypoint do projeto e não
  deve ser executado enquanto GigaVerbo estiver desabilitado;
- não existem configurações de execução, scripts de treino ou testes nesta
  entrega.
