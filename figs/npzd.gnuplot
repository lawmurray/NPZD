set terminal pdf enhanced dashed
#size 21cm,7.5cm
set output "figs/npzd.pdf"

#set tmargin 0
#set bmargin 0
#set lmargin 0
#set rmargin 0

set xtics scale 0 nomirror
set ytics scale 0 nomirror

set style line 1 linetype 1 linecolor rgb "#56B4E9" linewidth 2 pointtype 6
set style line 2 linetype 2 linecolor rgb "#E69F00" linewidth 2 pointtype 9
set style line 3 linetype 3 linecolor rgb "#009E73" linewidth 2 pointtype 6
set style line 4 linetype 4 linecolor rgb "#F0E442" linewidth 2 pointtype 9
set style line 5 linetype 5 linecolor rgb "#0072B2" linewidth 2 pointtype 6
set style line 6 linetype 6 linecolor rgb "#D55E00" linewidth 2 pointtype 9
set style line 7 linetype 7 linecolor rgb "#CC79A7" linewidth 2 pointtype 11

set grid

set style fill transparent solid 0.5 border
set style data lines

set xlabel "Days"

plot "results/simulate.csv" using 1:2 title "P" linestyle 1, \
"results/simulate.csv" using 1:3 title "Z" linestyle 2, \
"results/simulate.csv" using 1:4 title "D" linestyle 3, \
"results/simulate.csv" using 1:5 title "N" linestyle 4
