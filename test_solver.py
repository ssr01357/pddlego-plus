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

df = """(define (domain exploration)
  (:requirements :strips :negative-preconditions :typing)

  (:types
    receptacle
    object
    microwaveReceptacle sinkbasinReceptacle fridgeReceptacle - receptacle
    sharpobject - object
  )

  (:predicates
    (at ?r - receptacle)
    (visited ?r - receptacle)
    (opened ?r - receptacle)
    (has ?r - receptacle ?o - object)
    (sliced ?o - object)
    (used ?o - object)
    (heated ?o - object)
    (cleaned ?o - object)
    (cooled ?o - object)
  )

  (:action GotoLocation
    :parameters (?from - receptacle ?to - receptacle)
    :precondition (and (at ?from) (not (visited ?to)))
    :effect (and
      (not (at ?from))
      (at ?to)
      (visited ?to)
    )
  )

  (:action OpenObject
    :parameters (?r - receptacle)
    :precondition (and (at ?r) (not (opened ?r)))
    :effect (opened ?r)
  )

  (:action CloseObject
    :parameters (?r - receptacle)
    :precondition (opened ?r)
    :effect (not (opened ?r))
  )

  (:action PickupObject
    :parameters (?o - object ?r - receptacle)
    :precondition (and (at ?r) (has ?r ?o))
    :effect (not (has ?r ?o))
  )

  (:action PutObject
    :parameters (?o - object ?r - receptacle)
    :precondition (at ?r)
    :effect (has ?r ?o)
  )

  (:action UseObject
    :parameters (?o - object)
    :effect (used ?o)
  )

  (:action HeatObject
    :parameters (?o - object ?r - microwaveReceptacle)
    :precondition (at ?r)
    :effect (heated ?o)
  )

  (:action CleanObject
    :parameters (?o - object ?r - sinkbasinReceptacle)
    :precondition (at ?r)
    :effect (cleaned ?o)
  )

  (:action CoolObject
    :parameters (?o - object ?r - fridgeReceptacle)
    :precondition (at ?r)
    :effect (cooled ?o)
  )

  (:action SliceObject
    :parameters (?r - receptacle ?co - object ?sharp_o - sharpobject)
    :precondition (at ?r)
    :effect (sliced ?co)
  )
)
"""

pf = """(define (problem explore-room)
  (:domain exploration)

  (:objects
    init_receptacle
    cabinet1 cabinet2 cabinet3 cabinet4 cabinet5 cabinet6 cabinet7 cabinet8 cabinet9 cabinet10
    coffeemachine1
    countertop1 countertop2 countertop3
    diningtable1
    drawer1 drawer2 drawer3 drawer4 drawer5 drawer6
    stoveburner1 stoveburner2 stoveburner3 stoveburner4
    toaster1
    - receptacle

    microwave1
    - microwaveReceptacle

    sinkbasin1
    - sinkbasinReceptacle

    fridge1
    - fridgeReceptacle
  )

  (:init
    (at init_receptacle)
    (visited init_receptacle)
  )

  (:goal
    (at cabinet10)
  )
)
"""

print(run_solver(df, pf, "dual-bfws-ffparser"))
