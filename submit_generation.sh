#!/bin/bash

# Caveats:
# - module load needs to be run locally, before submitting
# - the info submitted to the amcatnlo prompt isn't being read correctly; the existing process_folder needs to be setup correctly before submitting (e.g. nevents, whether to run delphes, etc)

# --------------------------------------------------------------------- #
# Select partition, or queue. The LIP users should use "lipq" partition
#SBATCH -p lipq
# --------------------------------------------------------------------- #

parent_dir=$PWD
process_folder=MG5_aMC_v3_5_0/zh_mumu_bb_cut_ptmin_150

run_number=10
echo "This is the run_number ${run_number}"

echo "Checking if it already exists..."
CHECK=${parent_dir}/${process_folder}/Events/run_${run_number}
if [ -d "$CHECK" ];
then
   echo "$CHECK already exists. Exiting."
   exit 1
else
   echo "No, carrying on."
fi

#cp generation_opts.txt generation_opts_${run_number}.txt
#sed -i 's/PLACEHOLDER/${run_number}/' generation_opts_${run_number}.txt

module load gcc63/root/6.24.06

rm $process_folder/RunWeb
./$process_folder/bin/generate_events run_${run_number}
# <<EOF
#shower=Pythia8
#detector=OFF
#reweight=OFF
#madspin=ON
#EOF
#set nevents 10000<<EOF
#EOF

echo "DONE"
