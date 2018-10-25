#!/bin/bash


#set -f

target_username=$1
target_group=$1

if [ "$target_username" == "" ]; then
  echo "usage: $0 username"
  exit
fi

source_dir="/home/roger/notebooks/the_culture_of_international_relations"
target_dir="/home/${target_username}/notebooks/the_culture_of_international_relations"
backup_dir="/home/${target_username}/"

find $target_dir -name "*.py"  -o -name "*.ipynb" -o -name "*.md" | grep -v checkpoints | tar -cf ${backup_dir}/"$(date +"%Y%m%d_%H%M%S")".tar.gz -T -

#declare -a soft_link_dirs=("common" "corpora" "data" "images")
#for i in "${soft_link_dirs[@]}"
#do
#  if [ ! -L ${target_dir}/"$i" ]; then
#     sudo -u $target_username -g $target_group ln -s ${source_dir}/"$i" ${target_dir}/"$i"
#  fi
#done

if [ ! -d ${target_dir} ]; then
  sudo -u $target_username -g $target_group mkdir ${target_dir}
fi

declare -a notebook_dirs=("1_quantitative_analysis" "2_network_analysis" "3_text_analysis" "3_text_analysis/3.1_topic-model" "common" "common/network")

for i in "${notebook_dirs[@]}"
do
  if [ ! -d ${target_dir}/"$i" ]; then
    sudo -u $target_username -g $target_group mkdir ${target_dir}/"$i"
  fi
  rm -f ${target_dir}/${i}/*.*py*
  sudo -u $target_username -g $target_group cp ${source_dir}/${i}/*.*py* ${target_dir}/${i}/
done

