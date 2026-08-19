# WackyWacky PT-BR

Inventário de 2026-08-18 do arquivo `wacky/pages.tsv` em origem somente
leitura.

| Medida | Valor |
| --- | ---: |
| tamanho | 56.205.366.282 bytes |
| formato | TSV UTF-8 |
| colunas | 19 |
| total de linhas | desconhecido |

Uma amostra histórica limitada de 2.100 linhas em sete offsets encontrou 621
textos. Ela não é usada para estimar o rendimento porque a fonte parece
ordenada por status.

Campos utilizados na seleção: `domain_id`, `status`, `text`, `text_md5`,
`same_as`, `url`, `url_final` e `title`. URLs e títulos são consultados somente
em memória para rejeitar páginas de busca e listagem; não entram no texto, nos
relatórios ou nos manifests. HTML, timestamps e IDs também não são preservados.

Na fonte, `text` não está em texto puro: contém um frame Zstandard codificado
como hexadecimal. O adapter converte hexadecimal para bytes, descompacta ZSTD,
exige UTF-8 estrito e compara o conteúdo descompactado contra `text_md5` antes
da limpeza e tokenização. Divergências de MD5 observadas na fonte são mantidas
e contabilizadas em `text_md5_mismatches`; payloads inválidos ou maiores que
128 MiB são rejeitados sem registrar conteúdo.

## Seleção

São elegíveis apenas registros que satisfazem todas as regras:

```text
status == "done"
text presente
text_md5 presente
same_as ausente (campo vazio ou marcador literal "NULL")
página não identificada como busca, categoria, tag ou arquivo
```

Referências reais em `same_as`, como IDs numéricos, continuam excluídas. O
marcador textual `NULL` representa ausência na fonte e não é tratado como uma
referência.

O perfil `smoke` usa um prefixo limitado para engenharia. O `mvp` faz uma
passagem sequencial completa, calcula o fingerprint durante a leitura e mantém
um reservoir determinístico dentro do budget. O cursor por byte e por registro
permite retomar a passagem; qualquer alteração incompatível da fonte é
rejeitada.

## Boilerplate exato

No perfil `smoke`, a decisão é `remove_exact`: o filtro é aplicado e a execução
continua normalmente. No perfil `mvp`, a decisão inicial é `pending`; a execução
gera `boilerplate_report.json` e bloqueia a publicação dos shards finais. O
relatório v2 contém somente contagens agregadas, sem exemplos, hashes de blocos
ou conteúdo da fonte.

São detectadas duas classes conservadoras de repetição exata:

- parágrafos normalizados com pelo menos 80 caracteres, presentes em 5
  documentos e 3 domínios distintos;
- janelas de 3 linhas consecutivas, com pelo menos 60 caracteres combinados,
  presentes em 5 documentos distintos do mesmo domínio.

Antes da análise de repetição, cada linha não vazia é normalizada e removida se
tiver menos de 40 caracteres. O teste é aplicado separadamente depois de cada
quebra de linha e em todo o documento. A ordem e as separações entre os
parágrafos restantes são preservadas.

Repetições no mesmo documento contam como uma única presença. Blocos
sobrepostos marcam as linhas, que são removidas apenas uma vez. Não há busca
aproximada. Após a remoção, documentos afetados são descartados se restarem
menos de 300 caracteres ou se mais de 80% do texto normalizado tiver sido
removido.

Depois da revisão, altere `filters.boilerplate.decision_by_profile.mvp` em
`configs/datasets/wackywacky.json` para uma opção explícita:

- `keep`: preserva o texto dos candidatos;
- `remove_exact`: aplica apenas as duas regras exatas documentadas.

A mudança de `pending` para `keep` ou `remove_exact` reutiliza os candidatos
salvos e não relê a fonte completa. Alterações nos limiares invalidam esse cache
de análise e geram um novo `preparation-id`.

```sh
python -m queroquero.prepare run --dataset wackywacky --profile mvp
```

Deduplicação exata de documentos, limpeza, tokenização, split e packing são
responsabilidade do núcleo comum. A amostra histórica parcial não é usada para
estimar o rendimento global.
