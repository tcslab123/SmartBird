import pygame
import game
import training
import neural_network
import numpy as np  # For array manipulation
# import matplotlib.pyplot as plt # For visualizing images in array format

lives = 100
best_thought_processes = [neural_network.thought_process(0, 0, 0, 0, 0, 0), neural_network.thought_process(0, 0, 0, 0, 0, 0), neural_network.thought_process(0, 0, 0, 0, 0, 0)]
graphs = 0

def game_start(user_input):
    # GAME VARIABLES
    clock = pygame.time.Clock()
    floor = game.FLOOR
    bird = game.Bird(210,350)
    base = game.Base(floor)
    pipes = [game.Pipe(700)]
    win = game.win

    # NEURAL NETWORK VARIABLES
    model = neural_network.model(True, 0, 0, user_input)
    fitness_score = 0

    # Training parameters
    epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 32
    """ 
    NEURAL NETWORK VARIABLES
    """

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()
                break

        pipe_ind = 0
        game.draw_window(win, bird, pipes, base, pipe_ind)
        """
        NN CODE
        """
        state = training.preprocessing(pygame.surfarray.array3d(win))
        """
        NN CODE
        """
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1 

        if bird.y + bird.img.get_height() >= floor or bird.y < -220:
            run = False
            break

        bird.move()
      
        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            if pipe.collide(bird, win):
                run = False
                break
            
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                """ 
                NEURAL NETWORK TRAINING
                """
                # V careful, when using standard deviation, this should be a value that agrees
                fitness_score = fitness_score * 2
                """ 
                NEURAL NETWORK TRAINING
                """
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)
        
        """ 
        NEURAL NETWORK INPUT/OUTPUT
        """
        q_values = model.forward(state)
        if (q_values[0][0] > q_values[0][1]):
            bird.jump()
        """ 
        NEURAL NETWORK INPUT/OUTPUT
        """
        """ 
        NEURAL NETWORK TRAINING
        """
        fitness_score += 1
        """ 
        NEURAL NETWORK TRAINING
        """
        game.draw_window(win, bird, pipes, base, pipe_ind)
        next_state = training.preprocessing(pygame.surfarray.array3d(win))

    thought_process = neural_network.thought_process(fitness_score, model.hidden_layer.weights, model.hidden_layer.biases, model.output_layer.weights, model.output_layer.biases, user_input)
    print(thought_process[0][0])
    return thought_process

if __name__ == '__main__':
    user_input = game.menu_screen()
    # The total number of attempts the current generation of birds have
    for x in range(lives): 
        thought_process = game_start(user_input)
        # The total number of thought processes that I want to keep track of
        if thought_process[0][0] > best_thought_processes[0][0][0]:
            best_thought_processes[0], best_thought_processes[1], best_thought_processes[2] = thought_process, best_thought_processes[0], best_thought_processes[1]
        elif thought_process[0][0] > best_thought_processes[1][0][0]:
            best_thought_processes[1], best_thought_processes[2] = thought_process, best_thought_processes[1]
        elif thought_process[0][0] > best_thought_processes[2][0][0]:
            best_thought_processes[2] = thought_process
    print(best_thought_processes[0][0][0], best_thought_processes[1][0][0], best_thought_processes[2][0][0])
    print(best_thought_processes)