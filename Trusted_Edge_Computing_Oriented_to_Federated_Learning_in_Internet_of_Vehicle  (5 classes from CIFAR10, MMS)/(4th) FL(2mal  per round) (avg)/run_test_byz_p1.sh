#!/usr/bin/env bash

GPU=$1

DATASET=$2

NET=$3

BIAS=$4

NWORKERS=$5

BATCH_SIZE=$6

LR=$7

NITER=$8

SERVER_PC=$9

SEED=$10

NBYZ=$11

BYZ_TYPE=$12

AGGREGATION=$13

P=$14

TV1=$15

TV2=$16

TV3=$17

TV4=$18

TV5=$19

TV6=$20

TV7=$21

TV8=$22

TV9=$23

TV10=$24

python3 ./test_byz_p1.py \
--gpu $GPU \
--dataset $DATASET \
--net $NET \
--bias $BIAS \
--nworkers $NWORKERS \
--batch_size $BATCH_SIZE \
--lr $LR \
--niter $NITER \
--server_pc $SERVER_PC \
--nrepeats $SEED \
--nbyz $NBYZ \
--byz_type $BYZ_TYPE \
--aggregation $AGGREGATION \
--p $P \
--trust_value1 $TV1 \
--trust_value2 $TV2 \
--trust_value3 $TV3 \
--trust_value4 $TV4 \
--trust_value5 $TV5 \
--trust_value6 $TV6 \
--trust_value7 $TV7 \
--trust_value8 $TV8 \
--trust_value9 $TV9 \
--trust_value10 $TV10 \

