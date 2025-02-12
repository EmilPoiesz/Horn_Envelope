from sympy import *
import functools
import timeit
import Binary_parser

def evaluate(clause, x, V):
    """
    Evaluates the clause as True or False with given assignment x.

    Args:
    clause ():
        The clause that is to be evaluated.
    x (list):
        A binary lists where 1 correspond to a true variable and 0 to false.
    V (list):
        An exhaustive list of possible variables. 

    Returns:
    The truth value of the clause with the given assignment.
    """
    
    # Clause is a tautology or a contradiction
    if clause == True or clause == False: return clause 

    assignment = {V[i]: x[i] for i in range(len(V))}
    return clause.subs(assignment)

def models(assignment, hypothesis, V):
    """
    Checks if the given assignment models the current hypothesis. If there is a clause in the
    hypothesis that is evaluated to false by the assignment then the assignment does not model 
    the hypothesis.

    Args:
    assignment (list): 
        An binary list where 1 corresponds to a true variable and 0 corresponds to a false variable.
    hypothesis (set): 
        A set of clauses that together form the hypothesis.
    V (list):
        An exhaustive list of possible variables. 
    """
    for clause in hypothesis:
        if not evaluate(clause, assignment, V):
            return False
    return True

def is_subset(subset, superset):
    return all(x <= y for x, y in zip(subset, superset))

def intersection_of_lists(list_of_lists):
    return functools.reduce(lambda x, y: [a & b for a, b in zip(x, y)], list_of_lists)

def learn_horn_envelope(V:list, ask_membership_oracle, ask_equivalence_oracle, binary_parser:Binary_parser, background:set, verbose:bool=False, iterations:int=-1):
    
    metadata = []

    H = background.copy()
    Q = set()
    
    negative_counterexamples = []
    positive_counterexamples = []
    non_horn_counterexamples = []

    while iterations!=0:
        start = timeit.default_timer()
        iteration_data = {}
        
        # Issue with EQ oracle, checking H is fine, checking Q is not.
        equivalence_oracle_response = ask_equivalence_oracle(H.union(Q))

        if type(equivalence_oracle_response) == bool and equivalence_oracle_response == True:
            stop = timeit.default_timer()
            iteration_data['runtime'] = stop-start
            metadata.append(iteration_data)
            with open('output.txt', 'a') as f:
                f.write("=== TERMINATED ===\n")
            break
        
        (counterexample, sample_number) = equivalence_oracle_response
        iteration_data['sample'] = sample_number
        
        # Positive counterexample check
        positive_counterexample_flag = False
        for clause in H.union(Q):
            
            if evaluate(clause, counterexample, V): continue
            assert clause not in background
            
            positive_counterexamples.append(counterexample)
            positive_counterexample_flag = True
            break

        # Negative counterexample check
        if not positive_counterexample_flag:

            replaced_flag = False
            for neg_example in negative_counterexamples:

                counterexample_intersection = [neg_example[i] & counterexample[i] for i in range(len(V))]
                
                if (counterexample_intersection != neg_example) and \
                   (not ask_membership_oracle(counterexample_intersection)) and \
                    models(counterexample_intersection, Q, V):
                    
                    idx = negative_counterexamples.index(neg_example)
                    negative_counterexamples[idx] = counterexample_intersection
                    replaced_flag = True
                    break

            if not replaced_flag: negative_counterexamples.append(counterexample)

        # Non-Horn check
        for neg_counterexample in negative_counterexamples:
            positive_superset = [pos_counterexample for pos_counterexample in positive_counterexamples if is_subset(neg_counterexample, pos_counterexample)]
            if positive_superset == []: continue
            if neg_counterexample == intersection_of_lists(positive_superset):
                negative_counterexamples.remove(neg_counterexample)
                non_horn_counterexamples.append(neg_counterexample)
        
        # Reconstruct H
        H = background.copy()
        for neg_counterexample in negative_counterexamples:
            positive_superset = [pos_counterexample for pos_counterexample in positive_counterexamples if is_subset(neg_counterexample, pos_counterexample)]
            antecedent = And(*[V[i] for i, val in enumerate(neg_counterexample) if val == 1])
            
            if len(positive_superset) == 0: consequent = False
            else: consequent = And(*[V[i] for i, val in enumerate(intersection_of_lists(positive_superset)) if val == 1])
            
            implication = Implies(antecedent, consequent)
            H.add(implication)

        # Reconstruct Q
        Q = set()
        for nh_counterexample in non_horn_counterexamples:
            antecedent = And(*[V[i] for i, val in enumerate(nh_counterexample) if val == 1])
            consequent = Or(*[V[i] for i, val in enumerate(nh_counterexample) if val == 0])
            implication = Implies(antecedent, consequent)
            Q.add(implication)
        
        if verbose ==2:
            signed_counterexample = '+' if positive_counterexample_flag else '-'
            print(f'\nIteration: {abs(iterations)}\n\n' + 
                  f'({sample_number}) Counterexample: ({signed_counterexample}) {binary_parser.sentence_from_binary(counterexample)}\n\n'+
                  f'New Hypothesis H: {sorted([str(h) for h in H if h not in background])}\n\n' +
                  f'New Hypothesis Q: {[q for q in Q]}\n\n' +
                  f'New Hypothesis length: {len(H)+len(Q)-len(background)} + background: {len(background)}\n\n' +
                  f'total positive counterexamples:  {len(positive_counterexamples)}\n' +
                  f'total negative counterexamples:  {len(negative_counterexamples)} ({sum(sum(lst) for lst in negative_counterexamples)})\n' +
                  f'total non-horn counterexamples:  {len(non_horn_counterexamples)}\n\n\n\n')
        elif verbose == 1:
            print(f'Iteration {abs(iterations)}', end='\r')
        
        iterations-=1
        stop = timeit.default_timer()
        iteration_data['runtime'] = stop-start
        metadata.append(iteration_data)
        if iterations % 5 == 0:
            sentence = "iteration = {eq}\tlen(H) = {h}\truntime = {rt}\n".format(eq = 5000 - iterations, h=len(H.union(Q)), rt = iteration_data['runtime'])
            with open('output.txt', 'a') as f:
                f.write(sentence)
    
    terminated = iterations != 0  
    return (terminated, metadata, H, Q)
