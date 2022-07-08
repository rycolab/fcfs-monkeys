for seed in 00 01 02 03 04 05 06 07 08 09
do
    for lang in 'fi' he pt tr id en simple
    do
        jid2=$(SEED=${seed} LANGUAGE=${lang} sbatch scripts/hpc.wilkes2)
        jid2=$(echo $jid2 | cut -d' ' -f4)
        echo "Submitted " ${lang} ${jid2}
    done
done


