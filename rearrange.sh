# gdown --id 1Y8iJ5vr2sOKrJ-B7znrUAAzx0Gzj2bof
# tar -xzvf simmc.tar.gz && rm simmc.tar.gz

unziprm() {
    unzip $1.zip && rm $1.zip
}

cd data
unziprm simmc2_scene_images_dstc10_public_part1
unziprm simmc2_scene_images_dstc10_public_part2
unziprm simmc2_scene_jsons_dstc10_public

mkdir -p images 
mv simmc2_scene_images_dstc10_public_part1/* images
mv simmc2_scene_images_dstc10_public_part2/* images
rm -rf simmc2_scene_images_dstc10_public_part1
rm -rf simmc2_scene_images_dstc10_public_part2

mv public jsons
cd ..