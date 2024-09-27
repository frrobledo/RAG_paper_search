for file in ~/documents/PDF/*.pdf; do
    pdftotext "$file" "${file%.pdf}.txt"
done
