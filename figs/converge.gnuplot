set terminal pdf enhanced dashed
#size 21cm,7.5cm
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
set style line 4 linetype 4 linecolor rgb "#F0E442" linewidth 3 pointtype 4
set style line 5 linetype 5 linecolor rgb "#0072B2" linewidth 3 pointtype 5
set style line 6 linetype 6 linecolor rgb "#D55E00" linewidth 3 pointtype 6
set style line 7 linetype 7 linecolor rgb "#CC79A7" linewidth 3 pointtype 11
set style line 8 linetype 6 linecolor rgb "#56B4E9" linewidth 3 pointtype 1
set style line 9 linetype 5 linecolor rgb "#E69F00" linewidth 3 pointtype 2
set style line 10 linetype 4 linecolor rgb "#009E73" linewidth 3 pointtype 3
set style line 11 linetype 3 linecolor rgb "#F0E442" linewidth 3 pointtype 4
set style line 12 linetype 2 linecolor rgb "#0072B2" linewidth 3 pointtype 5

set grid

set style fill transparent solid 0.5 border
set style data lines

set key left top reverse Left
set xlabel "Step"
set ylabel "R_p"
set format x "10^{%L}"
#set format y "10^{%L}"
set xrange [ 500:25000 ]
set yrange [ 1:1.25 ]

set logscale x 10
#set logscale y 10

plot "results/dmcmc-share-16-converge.csv" title "DMCMC share 16" linestyle 1, \
"results/dmcmc-noshare-16-converge.csv" title "DMCMC no share 16" linestyle 4, \
"results/haario-16-converge.csv" title "Adaptive 16" linestyle 7, \
"results/straight-16-converge.csv" title "Random walk 16" linestyle 10

#plot "results/dmcmc-share-4-converge.csv" title "DMCMC share 4" linestyle 1, \
#"results/dmcmc-share-8-converge.csv" title "DMCMC share 8" linestyle 2, \
#"results/dmcmc-share-16-converge.csv" title "DMCMC share 16" linestyle 3, \
#"results/dmcmc-noshare-4-converge.csv" title "DMCMC no share 4" linestyle 4, \
#"results/dmcmc-noshare-8-converge.csv" title "DMCMC no share 8" linestyle 5, \
#"results/dmcmc-noshare-16-converge.csv" title "DMCMC no share 16" linestyle 6, \
#"results/haario-4-converge.csv" title "Adaptive 4" linestyle 7, \
#"results/haario-8-converge.csv" title "Adaptive 8" linestyle 8, \
#"results/haario-16-converge.csv" title "Adaptive 16" linestyle 9, \
#"results/straight-4-converge.csv" title "Random walk 4" linestyle 10, \
#"results/straight-8-converge.csv" title "Random walk 8" linestyle 11, \
#"results/straight-16-converge.csv" title "Random walk 16" linestyle 12
