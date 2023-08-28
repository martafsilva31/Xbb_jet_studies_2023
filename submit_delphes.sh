#!/bin/bash

# --------------------------------------------------------------------- #
# Select partition, or queue. The LIP users should use "lipq" partition
#SBATCH -p lipq
# --------------------------------------------------------------------- #

parent_dir=$PWD
process_folder=MG5_aMC_v3_5_0/zh_mumu_bb_cut_ptmin_150

run_number=06
echo "This is the run_number ${run_number}"

echo "Checking if it already exists..."
CHECK=${parent_dir}/${process_folder}/Events/run_${run_number}
if [ -d "$CHECK" ];
then
   echo "$CHECK is ready, will run Delphes on its output."
else
   echo "Run $run_number does not exist yet."
   exit 1
fi

module load gcc63/root/6.24.06

cd Delphes-3.5.0
#gzip -d ../MG5_aMC_v3_5_0/zh_mumu_bb_cut_ptmin_150/Events/run_${run_number}/tag_1_pythia8_events.hepmc.gz

./DelphesHepMC2 ../delphes_cards/delphes_card_ATLAS_PileUp.tcl delphes_${run_number}.root ../MG5_aMC_v3_5_0/zh_mumu_bb_cut_ptmin_150/Events/run_${run_number}/tag_1_pythia8_events.hepmc
mv delphes_${run_number}.root zh_mumu_bb_cut_ptmin_150_delphes_altVR_SHC_100mu
echo "DONE"
