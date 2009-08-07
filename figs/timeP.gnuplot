set terminal pdf enhanced dashed
#size 21cm,7.5cm
set output "figs/timeP.pdf"

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
set ylabel "Runtime ({/Symbol m}s)"
set format x "2^{%L}"
set format y "10^{%L}"

set logscale x 2
set logscale y 10

plot "results/timeP_DOPRI5_double.csv" using 1:2 title "GPU DOPRI5 double precision" linestyle 3, \
"results/timeP_DOPRI5_intrinsic.csv" using 1:2 title "GPU DOPRI5 single precision" linestyle 4, \
"results/timeP_RK43_double.csv" using 1:2 title "GPU RK4(3)5[2R+]C double precision" linestyle 5, \
"results/timeP_RK43_intrinsic.csv" using 1:2 title "GPU RK4(3)5[2R+]C single precision" linestyle 6
