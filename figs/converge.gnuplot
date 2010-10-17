set terminal pdf enhanced dashed size 21cm,7cm
set output "figs/converge.pdf"

#set tmargin 0
#set bmargin 0
#set lmargin 0
#set rmargin 0

set xtics scale 0 nomirror
set ytics scale 0 nomirror

set style line 1 linetype 1 linecolor rgb "#56B4E9" linewidth 3 pointtype 1
set style line 2 linetype 2 linecolor rgb "#E69F00" linewidth 3 pointtype 2
set style line 3 linetype 3 linecolor rgb "#009E73" linewidth 3 pointtype 3
set style line 4 linetype 4 linecolor rgb "#D55E00" linewidth 3 pointtype 4
set style line 5 linetype 5 linecolor rgb "#0072B2" linewidth 3 pointtype 5
set style line 6 linetype 6 linecolor rgb "#F0E442" linewidth 3 pointtype 6
set style line 7 linetype 7 linecolor rgb "#CC79A7" linewidth 3 pointtype 11

set grid

set style fill solid 0.8 border
set style data filledcurves

set key left top reverse Left
#set xlabel "Step"
set ylabel "R^p"
#set xrange [ 0:25000 ]
set yrange [ 1:1.5 ]
set key top right

set multiplot layout 1,4

plot \
"results/dmcmc-share-2-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 1, \
"" using 1:2 every 10 notitle with lines linestyle 1, \
"results/dmcmc-noshare-2-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 2, \
"" using 1:2 every 10 notitle with lines linestyle 2, \
"results/straight-2-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 4, \
"" using 1:2 every 10 notitle with lines linestyle 4

set ylabel ""
set format y ""

plot \
"results/dmcmc-share-4-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 1, \
"" using 1:2 every 10 notitle with lines linestyle 1, \
"results/dmcmc-noshare-4-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 2, \
"" using 1:2 every 10 notitle with lines linestyle 2, \
"results/straight-4-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 4, \
"" using 1:2 every 10 notitle with lines linestyle 4

plot \
"results/dmcmc-share-8-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 1, \
"" using 1:2 every 10 notitle with lines linestyle 1, \
"results/dmcmc-noshare-8-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 2, \
"" using 1:2 every 10 notitle with lines linestyle 2, \
"results/straight-8-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 4, \
"" using 1:2 every 10 notitle with lines linestyle 4

plot \
"results/dmcmc-share-16-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 1, \
"" using 1:2 every 10 title "Mixture, with sharing" with lines linestyle 1, \
"results/dmcmc-noshare-16-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 2, \
"" using 1:2 every 10 title "Mixture, without sharing" with lines linestyle 2, \
"results/straight-16-converge.csv" using 1:($2-$3):($2+$3) every 10 notitle linestyle 4, \
"" using 1:2 every 10 title "Random walk" with lines linestyle 4
