mkdir -p raw/yelp
mkdir raw/ridb
mkdir clean

for f in *.zip; do
    if [[ $f = "RIDBFullExport_v1.zip" ]]; then
        unzip $f -d raw/ridb
    else
        unzip $f -d raw/
    fi
done

tar -xf yelp_dataset_challenge_academic_dataset.tar --directory raw/yelp
