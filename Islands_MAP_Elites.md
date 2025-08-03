## Islands evolution with MAP-Elites

### Islands model
Islands model is a classic parallel genetic algorithm. Specifically, we split the population into $m$ separate groups, or islands. Each island is initialized with a copy of the user-provided initial program and is evolved separately. That is, whenever a prompt is required, we first **uniformly sample an island** and then sample $k$ programs from that island to build the prompt. The programs generated from the LLM based on that prompt will later be stored in the same island. After every $G$ generation of evolution, we discard all the programs from the $\frac{m}{2}$ islands whose best instances have the lowest score. Each of these islands is then seeded with a single program, obtained by first choosing one of the surviving $\frac{m}{2}$ islands uniformly at random, and then retrieving the highest-scoring program from that island (breaking ties in favour of older programs, i.e., when two or more programs have the same highest score, we use the program with lower generation). The evolutionary process is then restarted from this state, in which the reset islands contain one high-performing program each.

## MAP-Elites algorithm

### Signature of program
Within each island, we further cluster programs according to their signature. We define the signature of a program as the tuple containing the program’s scores on each of the inputs (e.g., the cap set size for each input $n$). Programs with the same signature are clustered together.

When sampling a program within an island, we first sample an island’s cluster, and then a program within that cluster.
### Sampling cluster
When sampling a cluster, we favor those with larger score values. Specifically, let $s_i$ denote the score of the $i$-th cluster, defined as an aggregation (include mean, max and median) of all the scores in the signature that characterizes that cluster. The probability $p_i$ of choosing cluster i is  $p_i = \frac{\exp (s_i/T_{cluster})}{\sum_{i′} \exp (s_{i′} / T_{cluster})}$ , $T_{cluster} = T_0 \cdot \left (1 − \frac{n \mod N}{N} \right )$, where $T_{cluster}$ is the temperature parameter, $n$ is the current number of programs in the island, and $T_0=0.1$ and $N=30,000$ are hyper-parameters. This approach is sometimes referred to as the Boltzmann selection procedure. 

### Sampling program within a cluster
When sampling a program within a cluster, we favor shorter programs. In particular, let $l_i$ denote the negative length of the $i$-th program within the chosen cluster (measured as the number of characters), and let $\hat{l}_i = \frac{l_i − \min_{i′} {l_{i′}}} {max_{i′} l_{i′} +10^{-6}}$. We set the probability of each program proportional to $\exp(\hat{l}_i/T_{program})$, where $T_{program}$ is a temperature hyperparameter.