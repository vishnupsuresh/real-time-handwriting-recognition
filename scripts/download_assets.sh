#!/usr/bin/env bash
# Downloads handwriting fonts (Google Fonts, OFL/Apache licensed) and an English
# word list. Kept out of git because they are third-party binaries/data.
# Run from the repo root:  bash scripts/download_assets.sh
set -e
B=https://raw.githubusercontent.com/google/fonts/main
mkdir -p fonts/print fonts/cursive data

get() {  # $1 = style folder, $2 = path inside google/fonts
  curl -sSf -o "fonts/$1/$(basename "$2")" "$B/$2" && echo "ok   $2" || echo "skip $2"
}

for p in ofl/indieflower/IndieFlower-Regular.ttf ofl/patrickhand/PatrickHand-Regular.ttf \
  ofl/shadowsintolight/ShadowsIntoLight.ttf ofl/kalam/Kalam-Regular.ttf \
  ofl/gloriahallelujah/GloriaHallelujah.ttf ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf \
  ofl/nothingyoucoulddo/NothingYouCouldDo.ttf ofl/reeniebeanie/ReenieBeanie.ttf \
  ofl/handlee/Handlee-Regular.ttf apache/justanotherhand/JustAnotherHand-Regular.ttf \
  apache/rocksalt/RockSalt-Regular.ttf apache/permanentmarker/PermanentMarker-Regular.ttf \
  ofl/zeyada/Zeyada.ttf ofl/coveredbyyourgrace/CoveredByYourGrace.ttf \
  apache/schoolbell/Schoolbell-Regular.ttf ofl/gochihand/GochiHand-Regular.ttf \
  ofl/neucha/Neucha.ttf apache/comingsoon/ComingSoon-Regular.ttf \
  ofl/waitingforthesunrise/WaitingfortheSunrise.ttf; do get print "$p"; done

for p in ofl/greatvibes/GreatVibes-Regular.ttf ofl/allura/Allura-Regular.ttf \
  ofl/sacramento/Sacramento-Regular.ttf ofl/marckscript/MarckScript-Regular.ttf \
  ofl/parisienne/Parisienne-Regular.ttf ofl/alexbrush/AlexBrush-Regular.ttf \
  ofl/pacifico/Pacifico-Regular.ttf apache/satisfy/Satisfy-Regular.ttf \
  ofl/cedarvillecursive/Cedarville-Cursive.ttf apache/homemadeapple/HomemadeApple-Regular.ttf \
  ofl/meddon/Meddon.ttf ofl/labelleaurore/LaBelleAurore.ttf ofl/mrdafoe/MrDafoe-Regular.ttf \
  apache/yellowtail/Yellowtail-Regular.ttf ofl/dawningofanewday/DawningofaNewDay.ttf \
  ofl/cookie/Cookie-Regular.ttf ofl/kristi/Kristi-Regular.ttf \
  ofl/herrvonmuellerhoff/HerrVonMuellerhoff-Regular.ttf ofl/qwigley/Qwigley-Regular.ttf \
  ofl/italianno/Italianno-Regular.ttf ofl/tangerine/Tangerine-Regular.ttf \
  ofl/kaushanscript/KaushanScript-Regular.ttf ofl/courgette/Courgette-Regular.ttf \
  ofl/norican/Norican-Regular.ttf ofl/euphoriascript/EuphoriaScript-Regular.ttf \
  ofl/mrssaintdelafield/MrsSaintDelafield-Regular.ttf; do get cursive "$p"; done

curl -sSf -o data/words.txt \
  https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt
echo "fonts: $(ls fonts/print | wc -l) print, $(ls fonts/cursive | wc -l) cursive; words: $(wc -l < data/words.txt)"
