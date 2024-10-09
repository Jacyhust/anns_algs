while true; do
    (date && free -h)>> memory1.txt
    sleep 60  # Adjust the interval as needed
done
