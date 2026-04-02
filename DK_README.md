# Command to running the software: 
- open ./build/build/Release/XDAQ-RHX.app

# RHX TCP Test program: 
- python rhx_tcp_spike_client.py --channel A-000 --duration-s 60 --cleanup


# Pong Game 
- Live neural input: python pong_game.py --mode rhx
- Play against computer: python pong_game.py --mode keyboard

# Running the PONG game
- python pong_game.py --mode rhx --threshold 0.5 --smooth-windows 1

