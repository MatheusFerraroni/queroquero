# GigaVerbo-v2

Inventário de metadados oficiais de 2026-08-19. Nenhum registro individual foi
copiado para a documentação.

## Fonte ativa para preparação

| Campo | Valor |
| --- | --- |
| dataset | `Polygl0t/gigaverbo-v2` |
| revisão pinada | `b39dfa703102a20dc609ed6e7aaae22e8e3a233f` |
| configuração/split | `default` / `train` |
| acesso | público, streaming |
| licença agregada | `other` |
| política de derivados | `internal_research_only` |

| Configuração | Linhas | Bytes | Tokens declarados | Shards observados |
| --- | ---: | ---: | ---: | ---: |
| `default` | 372.108.576 | 834.750.093.054 | 317.688.116.144 | 224 |
| `excluded` | 2.892.095 | 7.894.131.263 | 2.987.598.133 | 23 |

Somente `default/train` foi selecionado. A lista de shards é resolvida e
manifestada na revisão pinada; `excluded` não é lido.

O card já mencionou 103 shards `default`, enquanto a revisão pinada contém 224;
por isso o adapter resolve a lista efetiva da revisão. Uma estatística parcial
histórica encontrou 3,36% de registros com score `>= 4`, mas não representa o
corpus completo.

## Adapter

O adapter usa streaming, embaralhamento determinístico com buffer finito e
aceita apenas registros com `edu_int_score >= 4`. A leitura para exatamente no
limite configurado de candidatos ou registros de origem. O texto é então
retokenizado com o tokenizer pinado do `Tucano2-0.6B-Base`.

`token_count` da fonte usa outro tokenizer e não define o budget. IDs dos
registros não são preservados; a proveniência usa uma referência derivada por
hash e metadados de origem estritamente limitados.

O perfil `smoke` processa no máximo 8.192 registros de origem para obter até
128 documentos candidatos. O `mvp` processa no máximo 262.144 registros para
obter até 4.096 candidatos. Os budgets finais continuam sendo 8+2 e 256+32
sequências de 1.024 tokens, respectivamente.

```sh
python -m queroquero.prepare run --dataset gigaverbo --profile smoke
```

Retomada usa o índice do stream e só é aceita para a mesma revisão. A preparação
requer rede e não baixa o corpus completo.

## Restrição de uso

O corpus combina fontes com licenças e termos diferentes. Sua preparação está
ativa apenas para pesquisa interna: dados, shards e quaisquer pesos futuros
permanecem `internal_research_only`. Este adapter não habilita treinamento nem
redistribuição.

## Referências

- [Dataset card](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2)
- [Revisão pinada](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2/tree/b39dfa703102a20dc609ed6e7aaae22e8e3a233f)
- [Tamanhos oficiais](https://datasets-server.huggingface.co/size?dataset=Polygl0t%2Fgigaverbo-v2)
