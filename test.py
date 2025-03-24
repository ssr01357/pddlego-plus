# import requests

# def run_solver(domain_file, problem_file, solver):

#     req_body = {"domain" : domain_file, "problem" : problem_file}

#     # Send job request to solve endpoint
#     solve_request_url=requests.post(f"https://solver.planning.domains:5001/package/{solver}/solve", json=req_body).json()

#     # Query the result in the job
#     celery_result=requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])

#     while celery_result.json().get("status","")== 'PENDING':
#         # Query the result every 0.5 seconds while the job is executing
#         celery_result=requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])

#     result = celery_result.json()['result']
#     return result

# df = """(define (domain exploration)
#     (:requirements :strips :typing)
#     (:types
#         location
#         direction
#     )
#     (:predicates
#         (door-open ?loc1 ?loc2 ?dir)
#         (visited ?loc)
#         (at ?loc)
#     )
#     (:action open-door
#     :parameters (?loc1 - location ?loc2 - location ?dir - direction)
#     :precondition (and (at ?loc1) (not (door-open ?loc1 ?loc2 ?dir)))
#     :effect (door-open ?loc1 ?loc2 ?dir)
#     )

#     (:action move
#     :parameters (?from - location ?to - location ?dir - direction)
#     :precondition (and (at ?from) (door-open ?from ?to ?dir))
#     :effect (and (at ?to) (not (at ?from)) (visited ?to))
# )"""

# pf = """(define (problem exploration-problem)
#     (:domain exploration)
#     (:objects
#         kitchen - location
#         south-room - location
#         west-room - location
#         south - direction
#         west - direction
#     )
#     (:init
#         (at kitchen)
#         (visited kitchen)
#         (not (door-open kitchen south-room south))
#         (not (door-open kitchen west-room west))
#     )
#     (:goal
#         (at west-room)
#     )
# )"""

# print(run_solver(df, pf, "dual-bfws-ffparser"))

import re

def map_actions(action): # (GOTOLOCATION AGENT1 NEW_LOCATION TOWELHOLDER1)\n
    actions = action.lower().replace("(", "").replace(")", "").split('\n')
    action_lst = []
    for act in actions:
        if "gotolocation" in act:
            location = act.split(' ')[-1]
            # Insert a space between non-digits and digits, e.g., "towelholder1" -> "towelholder 1"
            formatted_location = re.sub(r"(\D+)(\d+)", r"\1 \2", location)
            action_lst.append(f"go to {formatted_location}")
    if len(action_lst) == 0:
        return None
    return action_lst

print(map_actions("(GOTOLOCATION AGENT1 NEW_LOCATION TOWELHOLDER1)\n")) # ['go to towelholder 1']