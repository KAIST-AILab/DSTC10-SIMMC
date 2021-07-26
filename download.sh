# gdown --id 1Y8iJ5vr2sOKrJ-B7znrUAAzx0Gzj2bof
# tar -xzvf simmc.tar.gz && rm simmc.tar.gz

cd data
unzip simmc2_scene_images_dstc10_public_part1.zip && rm simmc2_scene_images_dstc10_public_part1.zip
unzip simmc2_scene_images_dstc10_public_part2.zip && rm simmc2_scene_images_dstc10_public_part2.zip

mkdir -p images 
mv simmc2_scene_images_dstc10_public_part1/* images
mv simmc2_scene_images_dstc10_public_part2/* images
rm -rf simmc2_scene_images_dstc10_public_part1
rm -rf simmc2_scene_images_dstc10_public_part2

unzip simmc2_scene_jsons_dstc10_public.zip && rm simmc2_scene_jsons_dstc10_public.zip

mv public jsons
cd ..