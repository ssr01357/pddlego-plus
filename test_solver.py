import requests


def run_solver(domain_file, problem_file, solver):

    req_body = {"domain" : domain_file, "problem" : problem_file}

    # Send job request to solve endpoint
    solve_request_url=requests.post(f"https://solver.planning.domains:5001/package/{solver}/solve", json=req_body).json()

    # Query the result in the job
    celery_result=requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])

    while celery_result.json().get("status","")== 'PENDING':
        # Query the result every 0.5 seconds while the job is executing
        celery_result=requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])

    result = celery_result.json()['result']
    return result

df = """(define (domain room)
  (:requirements :typing)
  (:types receptacle object microwaveReceptacle sinkbasinReceptacle fridgeReceptacle sharpobject - object)

  (:predicates
    (at ?r - receptacle)
    (opened ?r - receptacle)
    (pickedup ?o - object)
    (in ?o - object ?r - receptacle)
    (used ?o - object)
    (heated ?o - object)
    (cleaned ?o - object)
    (cooled ?o - object)
    (sliced ?o - object)
  )

  (:action GotoLocation
    :parameters (?from - receptacle ?to - receptacle)
    :precondition (at ?from)
    :effect (and (at ?to) (not (at ?from)))
  )

  (:action OpenObject
    :parameters (?r - receptacle)
    :precondition (at ?r)
    :effect (opened ?r)
  )

  (:action CloseObject
    :parameters (?r - receptacle)
    :precondition (at ?r)
    :effect (not (opened ?r))
  )

  (:action PickupObject
    :parameters (?o - object ?r - receptacle)
    :precondition (at ?r)
    :effect (and (pickedup ?o) (not (in ?o ?r)))
  )

  (:action PutObject
    :parameters (?o - object ?r - receptacle)
    :precondition (at ?r)
    :effect (in ?o ?r)
  )

  (:action useObject
    :parameters (?o - object)
    :precondition (pickedup ?o)
    :effect (used ?o)
  )

  (:action HeatObject
    :parameters (?o - object ?r - microwaveReceptacle)
    :precondition (and (at ?r) (in ?o ?r))
    :effect (heated ?o)
  )

  (:action CleanObject
    :parameters (?o - object ?r - sinkbasinReceptacle)
    :precondition (and (at ?r) (in ?o ?r))
    :effect (cleaned ?o)
  )

  (:action CoolObject
    :parameters (?o - object ?r - fridgeReceptacle)
    :precondition (and (at ?r) (in ?o ?r))
    :effect (cooled ?o)
  )

  (:action SliceObject
    :parameters (?loc - receptacle ?co - object ?sharp_o - sharpobject)
    :precondition (at ?loc)
    :effect (sliced ?co)
  )
) 
"""

pf = """(define (problem room-prob)
  (:domain room)
  (:objects
    init_receptacle - receptacle
    cabinet21 cabinet20 cabinet19 cabinet18 cabinet17 cabinet16 cabinet15 cabinet14 cabinet13 cabinet12 cabinet11 cabinet10 cabinet9 cabinet8 cabinet7 cabinet6 cabinet5 cabinet4 cabinet3 cabinet2 cabinet1 - receptacle
    countertop2 countertop1 - receptacle
    diningtable1 - receptacle
    drawer5 drawer4 drawer3 drawer2 drawer1 - receptacle
    fridge1 - fridgeReceptacle
    garbagecan1 - receptacle
    microwave1 - microwaveReceptacle
    sinkbasin1 - sinkbasinReceptacle
    stoveburner4 stoveburner3 stoveburner2 stoveburner1 - receptacle
    coffeemachine1 toaster1 - object
  )
  (:init
    (at init_receptacle)
  )
  (:goal (at cabinet21))
) 
"""

print(run_solver(df, pf, "dual-bfws-ffparser"))
