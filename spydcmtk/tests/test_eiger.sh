#!/bin/bash


# Run in DEV env
which python | grep '/envs/dev/bin/python' &> /dev/null
if [ $? != 0 ]; then
   echo "ERROR: Should be in 'envDEV' environment"
   exit 1
fi


spydcmtk -h

ROOTDIR=/DATA/TEST_DICOMS
OUTDIR=$ROOTDIR/TEMP_OUT
RAWDIR=$ROOTDIR/RAW_DATA
SE3=$RAWDIR/SE3_AX_FIESTA
SE4D=$RAWDIR/SE23_Ax_4DFLOW_1_slabC_-_Preview
RAWZIP=$ROOTDIR/RAW_DATA.zip
RAWTAR=$ROOTDIR/RAW_DATA.tar.gz
mkdir $OUTDIR

spydcmtk -i $SE4D -o $OUTDIR
spydcmtk -i $SE3 -o $OUTDIR -nii
spydcmtk -i $SE3 -o $OUTDIR -html
spydcmtk -i $RAWDIR -o $OUTDIR -dcmdump
spydcmtk -i $RAWDIR -o $OUTDIR -STREAM
spydcmtk -i $RAWDIR -o $OUTDIR -quickInspect
spydcmtk -i $RAWDIR -o $OUTDIR -quickInspectFull
spydcmtk -i $SE3 -o $OUTDIR -a NEW_ANON
spydcmtk -i $SE3 -o $OUTDIR -a OTHER_ANON -aid 123455
spydcmtk -i $SE3 -o $OUTDIR -msTable
spydcmtk -i $SE3 -o $OUTDIR -SAFE
spydcmtk -i $SE3 -o $OUTDIR -SAFE -FORCE
spydcmtk -i $SE3 -o $OUTDIR -SAFE -QUIET
spydcmtk -i $SE3 -o $OUTDIR -SAFE -STREAM -FORCE -QUIET

mkdir $OUTDIR/ZIP
spydcmtk -i $RAWZIP -o $OUTDIR/ZIP -SAFE

mkdir $OUTDIR/TAR
spydcmtk -i $RAWTAR -o $OUTDIR/TAR -SAFE

rm -r $OUTDIR