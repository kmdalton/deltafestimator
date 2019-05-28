pdflatex --interaction=batchmode main.tex
evince main.pdf&
sleep 5

while true;do 
  pdflatex --interaction=batchmode main.tex
  sleep 5
done
