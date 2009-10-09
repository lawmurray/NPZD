set terminal pdf enhanced dashed
#size 21cm,7.5cm
set output "figs/timeFilter.pdf"

#set tmargin 0
#set bmargin 0
#set lmargin 0
#set rmargin 0

set xtics scale 0 nomirror
set ytics scale 0 nomirror

set style line 1 linetype 1 linecolor rgb "#56B4E9" linewidth 3 pointtype 6
set style line 2 linetype 1 linecolor rgb "#E69F00" linewidth 3 pointtype 9
set style line 3 linetype 2 linecolor rgb "#009E73" linewidth 3 pointtype 6
set style line 4 linetype 2 linecolor rgb "#F0E442" linewidth 3 pointtype 9
set style line 5 linetype 4 linecolor rgb "#0072B2" linewidth 3 pointtype 6
set style line 6 linetype 4 linecolor rgb "#D55E00" linewidth 3 pointtype 9
set style line 7 linetype 7 linecolor rgb "#CC79A7" linewidth 3 pointtype 11

set grid

set style fill transparent solid 0.5 border
set style data linespoints

set key left top reverse Left
set xlabel "No. trajectories (P)"
set ylabel "Wallclock time (s)"
set xtics 1024

plot "results/timeFilter_double.csv" using 1:($2/1e6) title "Double precision" linestyle 1, \
"results/timeFilter_single.csv" using 1:($2/1e6) title "Single precision" linestyle 2

