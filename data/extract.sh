mkdir -p raw/yelp
mkdir clean

for f in *.zip; do
    unzip $f -d raw/;
done

tar -xf yelp_dataset_challenge_academic_dataset.tar --directory raw/yelp
