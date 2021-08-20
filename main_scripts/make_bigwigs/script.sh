bed_file=chr20.merged.overlap.bed
#CUDA_VISIBLE_DEVICES=2 python score_model_overlap_merged.py --weights /srv/scratch/anusri/chrombpnet_paper/GM12878/ATAC_07.22.2021/final_model_step3_new/model.0.weights --json_string /srv/scratch/anusri/chrombpnet_paper/GM12878/ATAC_07.22.2021/final_model_step3_new/model.0.arch --bigwig_labels /srv/scratch/anusri/chrombpnet_paper/GM12878/data/shifted_4_4.sorted.bam.bpnet.unstranded.bw --out_prefix gm12878_atac_uncorrected/atac --bed_file_to_score $bed_file --presorted_intervals --precentered_intervals  --chrom_sizes hg38.chrom.sizes
#patht=/srv/scratch/anusri/chrombpnet_paper/GM12878/ATAC_07.22.2021/final_model_step3_new/unplug
#CUDA_VISIBLE_DEVICES=0 python score_model_overlap_merged.py --weights /srv/scratch/anusri/chrombpnet_paper/GM12878/ATAC_07.22.2021/invivo_bias_model_step1/model.0.weights --json_string /srv/scratch/anusri/chrombpnet_paper/GM12878/ATAC_07.22.2021/invivo_bias_model_step1/model.0.arch --bigwig_labels /srv/scratch/anusri/chrombpnet_paper/GM12878/data/shifted_4_4.sorted.bam.bpnet.unstranded.bw --out_prefix invivo_bias/atac --bed_file_to_score $bed_file --presorted_intervals --precentered_intervals  --chrom_sizes hg38.chrom.sizes
#CUDA_VISIBLE_DEVICES=1 python score_model_overlap_merged.py --weights /srv/scratch/anusri/chrombpnet_paper/GM12878/ATAC_07.22.2021/bias_fit_on_signal_step2/model.0.weights --json_string /srv/scratch/anusri/chrombpnet_paper/GM12878/ATAC_07.22.2021/bias_fit_on_signal_step2/model.0.arch --bigwig_labels /srv/scratch/anusri/chrombpnet_paper/GM12878/data/shifted_4_4.sorted.bam.bpnet.unstranded.bw --out_prefix gm12878_atac_bias/atac --bed_file_to_score $bed_file --presorted_intervals --precentered_intervals --chrom_sizes hg38.chrom.sizes
#CUDA_VISIBLE_DEVICES=3 python score_model_overlap_merged.py --model_path /srv/scratch/anusri/chrombpnet_paper/GM12878/ATAC_07.22.2021/final_model_step3_new/unplug/model.0.hdf5 --bigwig_labels /srv/scratch/anusri/chrombpnet_paper/GM12878/data/shifted_4_4.sorted.bam.bpnet.unstranded.bw --out_prefix gm12878_atac_profile_and_count_corrected/atac --bed_file_to_score $bed_file --presorted_intervals --precentered_intervals --chrom_sizes hg38.chrom.sizes
#CUDA_VISIBLE_DEVICES=0 python tobias_score_model_overlap_merged.py --weights /srv/scratch/anusri/chrombpnet_paper/tobias_scripts/GM12878/tobias_ATAC_08.03.2021/final_model/model.0.weights --json_string /srv/scratch/anusri/chrombpnet_paper/tobias_scripts/GM12878/tobias_ATAC_08.03.2021/final_model/model.0.arch --bigwig_labels /srv/scratch/anusri/chrombpnet_paper/GM12878/data/shifted_4_4.sorted.bam.bpnet.unstranded.bw --out_prefix tobias_model_profile_corrected/atac --bed_file_to_score $bed_file --presorted_intervals --precentered_intervals  --chrom_sizes hg38.chrom.sizes
CUDA_VISIBLE_DEVICES=0 python tobias_score_model_overlap_merged.py --model_path /srv/scratch/anusri/chrombpnet_paper/tobias_scripts/GM12878/tobias_ATAC_08.03.2021/final_model/unplug/model.0.hdf5 --bigwig_labels /srv/scratch/anusri/chrombpnet_paper/GM12878/data/shifted_4_4.sorted.bam.bpnet.unstranded.bw --out_prefix tobias_model_profile_corrected/atac --bed_file_to_score $bed_file --presorted_intervals --precentered_intervals  --chrom_sizes hg38.chrom.sizes
#CUDA_VISIBLE_DEVICES=1 python score_model_overlap_merged.py --weights /srv/scratch/anusri/chrombpnet_paper/hint_atac_scripts/GM12878/hintatac_ATAC_07.27.2021/model/model.0.weights --json_string /srv/scratch/anusri/chrombpnet_paper/hint_atac_scripts/GM12878/hintatac_ATAC_07.27.2021/model/model.0.arch --bigwig_labels /srv/scratch/anusri/chrombpnet_paper/GM12878/data/shifted_4_4.sorted.bam.bpnet.unstranded.bw --out_prefix hintatac_corrected/atac --bed_file_to_score $bed_file --presorted_intervals --precentered_intervals  --chrom_sizes hg38.chrom.sizes
CUDA_VISIBLE_DEVICES=1 python tobias_score_model_overlap_merged.py --weights /srv/scratch/anusri/chrombpnet_paper/tobias_scripts/GM12878/tobias_ATAC_08.03.2021/final_model/model.0.weights --json_string /srv/scratch/anusri/chrombpnet_paper/tobias_scripts/GM12878/tobias_ATAC_08.03.2021/final_model/model.0.arch --bigwig_labels /srv/scratch/anusri/chrombpnet_paper/GM12878/data/shifted_4_4.sorted.bam.bpnet.unstranded.bw --out_prefix tobias_model_profile_corrected/atac --bed_file_to_score $bed_file --presorted_intervals --precentered_intervals --chrom_sizes hg38.chrom.sizes













