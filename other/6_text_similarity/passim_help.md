
## Install and Run PASSIM

PASSIM requires Java 1.8+ installed on the system.

```bash

% sudo apt-get install maven
% # export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"

% wget https://www.apache.org/dyn/closer.lua/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz
% mv spark-2.4.0-bin-hadoop2.7 /usr/local/share/spar
 
# Add to path: PATH="$HOME/source/passm/bin/:$PATH"
# Add to path: PATH="$PATH:/usr/local/share/spark/bin"

```

```bash
Usage: passim [options] <path>,<path>,... <path>

  --boilerplate                   Detect boilerplate within groups.
  -n, --n <value>                 index n-gram features; default=5
  -l, --minDF <value>             Lower limit on document frequency; default=2
  -u, --maxDF <value>             Upper limit on document frequency; default=100
  -m, --min-match <value>         Minimum number of n-gram matches between documents; default=5
  -a, --min-align <value>         Minimum length of alignment; default=20
  -g, --gap <value>               Minimum size of the gap that separates passages; default=100
  -c, --context <value>           Size of context for aligned passages; default=0
  -o, --relative-overlap <value>  Minimum relative overlap to merge passages; default=0.8
  -M, --merge-diverge <value>     Maximum length divergence for merging extents; default=0.3
  -r, --max-repeat <value>        Maximum repeat of one series in a cluster; default=10
  -p, --pairwise                  Output pairwise alignments
  -d, --docwise                   Output docwise alignments
  -N, --names                     Output names and exit
  -P, --postings                  Output postings and exit
  -i, --id <value>                Field for unique document IDs; default=id
  -t, --text <value>              Field for document text; default=text
  -s, --group <value>             Field to group documents into series; default=series
  -f, --filterpairs <value>       Constraint on posting pairs; default=gid < gid2
  --fields <value>                Semicolon-delimited list of fields to index
  --input-format <value>          Input format; default=json
  --output-format <value>         Output format; default=json
  -w, --word-length <value>       Minimum average word length to match; default=2
  --help                          prints usage text
  <path>,<path>,...               Comma-separated input paths
  <path>                          Output path
```