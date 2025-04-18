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

df = """(define (domain environment)
  (:requirements :typing)
  (:types receptacle object microwaveReceptacle fridgeReceptacle sinkbasinReceptacle sharpObject - object)
  (:predicates
    (at ?r - receptacle)
    (opened ?r - receptacle)
    (closed ?r - receptacle)
    (contains ?r - receptacle ?o - object)
  )
  (:action GotoLocation
    :parameters (?from - receptacle ?to - receptacle)
    :precondition (at ?from)
    :effect (and (not (at ?from)) (at ?to))
  )
  (:action OpenObject
    :parameters (?r - receptacle)
    :precondition (closed ?r)
    :effect (opened ?r)
  )
  (:action CloseObject
    :parameters (?r - receptacle)
    :precondition (opened ?r)
    :effect (closed ?r)
  )
  (:action PickupObject
    :parameters (?o - object ?r - receptacle)
    :precondition (contains ?r ?o)
    :effect (not (contains ?r ?o))
  )
  (:action PutObject
    :parameters (?o - object ?r - receptacle)
    :precondition (at ?r)
    :effect (contains ?r ?o)
  )
  (:action useObject
    :parameters (?o - object)
    :precondition (and)
    :effect (and)
  )
  (:action HeatObject
    :parameters (?o - object ?r - microwaveReceptacle)
    :precondition (contains ?r ?o)
    :effect ()
  )
  (:action CleanObject
    :parameters (?o - object ?r - sinkbasinReceptacle)
    :precondition (contains ?r ?o)
    :effect ()
  )
  (:action CoolObject
    :parameters (?o - object ?r - fridgeReceptacle)
    :precondition (contains ?r ?o)
    :effect ()
  )
  (:action SliceObject
    :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)
    :precondition (and (contains ?r ?co) (contains ?r ?sharp_o))
    :effect ()
  )
)"""

pf = """(define (problem room-problem)
  (:domain environment)
  (:objects
    init_receptacle - receptacle
    cabinet1 cabinet2 cabinet3 cabinet4 cabinet5 cabinet6 cabinet7 cabinet8 cabinet9 cabinet10
    cabinet11 cabinet12 cabinet13 cabinet14 cabinet15 cabinet16 cabinet17 cabinet18 cabinet19 cabinet20 - receptacle
    coffeemachine1 - object
    countertop1 countertop2 - receptacle
    diningtable1 diningtable2 - receptacle
    drawer1 drawer2 drawer3 drawer4 drawer5 drawer6 - receptacle
    fridge1 - fridgeReceptacle
    garbagecan1 - receptacle
    microwave1 - microwaveReceptacle
    sinkbasin1 - sinkbasinReceptacle
    stoveburner1 stoveburner2 stoveburner3 stoveburner4 - receptacle
    toaster1 - object
    knife1 - sharpObject
  )
  (:init
    (at init_receptacle)
    (closed cabinet1) (closed cabinet2) (closed cabinet3) (closed cabinet4) (closed cabinet5) (closed cabinet6) (closed cabinet7) (closed cabinet8) (closed cabinet9) (closed cabinet10)
    (closed cabinet11) (closed cabinet12) (closed cabinet13) (closed cabinet14) (closed cabinet15) (closed cabinet16) (closed cabinet17) (closed cabinet18) (closed cabinet19) (closed cabinet20)
    (closed drawer1) (closed drawer2) (closed drawer3) (closed drawer4) (closed drawer5) (closed drawer6)
  )
  (:goal (at cabinet1))
)"""

print(run_solver(df, pf, "dual-bfws-ffparser"))
