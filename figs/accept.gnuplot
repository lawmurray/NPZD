set terminal pdf enhanced dashed size 21cm,7cm
set output "figs/accept.pdf"

#set tmargin 0
#set bmargin 0
#set lmargin 0
#set rmargin 0

set xtics scale 0 nomirror
set ytics scale 0 nomirror

set style line 1 linetype 1 linecolor rgb "#56B4E9" linewidth 3 pointtype 1
set style line 2 linetype 2 linecolor rgb "#E69F00" linewidth 3 pointtype 2
set style line 3 linetype 3 linecolor rgb "#F0E442" linewidth 3 pointtype 3
set style line 4 linetype 4 linecolor rgb "#D55E00" linewidth 3 pointtype 4
set style line 5 linetype 5 linecolor rgb "#0072B2" linewidth 3 pointtype 5
set style line 6 linetype 6 linecolor rgb "#009E73" linewidth 3 pointtype 6
set style line 7 linetype 7 linecolor rgb "#CC79A7" linewidth 3 pointtype 11

set grid

set style fill transparent solid 0.5 border
set style data lines

set key left top reverse Left
#set xlabel "Step"
set ylabel "Log-accept"
#set format x "10^{%L}"
#set format y "10^{%L}"

#set logscale x 10
#set logscale y 10
set xrange [0:25000]
set yrange [-10:0]

set multiplot layout 1,4

plot "results/dmcmc-share-2-accept.csv" using 1:2 notitle linestyle 1, \
"results/dmcmc-noshare-2-accept.csv" using 1:2 notitle linestyle 2

set ylabel ""
set format y ""

plot "results/dmcmc-share-4-accept.csv" using 1:2 notitle linestyle 1, \
"results/dmcmc-noshare-4-accept.csv" using 1:2 notitle linestyle 2

plot "results/dmcmc-share-8-accept.csv" using 1:2 notitle linestyle 1, \
"results/dmcmc-noshare-8-accept.csv" using 1:2 notitle linestyle 2

plot "results/dmcmc-share-16-accept.csv" using 1:2 title "Mixture, with sharing" linestyle 1, \
"results/dmcmc-noshare-16-accept.csv" using 1:2 title "Mixture, without sharing" linestyle 2
