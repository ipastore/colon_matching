#!/bin/bash

# Main repository
cd /Users/ignaciopastorebenaim/Documents/MGRCV/COLON/colon_matching
git checkout feature/mask
git pull origin feature/mask

# Submodule: image-matching-models
cd utils/image-matching_models
git checkout feature/mask
git pull origin feature/mask

# Sub-submodule: LightGlue
cd matching/third_party/LightGlue
git checkout feature/lightglue_mask
git pull origin feature/lightglue_mask

# Sub-submodule: RoMa
cd ../RoMa
git checkout feature/TinyRoMa_mask
git pull origin feature/TinyRoMa_mask

echo "All submodules are updated!"