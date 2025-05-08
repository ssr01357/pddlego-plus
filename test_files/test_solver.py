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

df = """(define (domain room_task)
  (:requirements :typing :negative-preconditions)
  (:types 
    object
    receptacle
    fridgeReceptacle microwaveReceptacle sinkbasinReceptacle - receptacle
    sharpObject - object
  )
  (:predicates
    (at ?r - receptacle)
    (visited ?r - receptacle)
    (opened ?r - receptacle)
    (contains ?r - receptacle ?o - object)
    (inhand ?o - object)
    (used ?o - object)
    (heated ?o - object)
    (cleaned ?o - object)
    (cooled ?o - object)
    (sliced ?o - object)
  )

  (:action GotoLocation
    :parameters (?from - receptacle ?to - receptacle)
    :precondition (and (at ?from) (not (visited ?to)))
    :effect (and (not (at ?from)) (at ?to) (visited ?to))
  )

  (:action OpenObject
    :parameters (?r - receptacle)
    :precondition (and (at ?r) (not (opened ?r)))
    :effect (opened ?r)
  )

  (:action CloseObject
    :parameters (?r - receptacle)
    :precondition (at ?r)
    :effect (not (opened ?r))
  )

  (:action PickupObject
    :parameters (?o - object ?r - receptacle)
    :precondition (and (at ?r) (contains ?r ?o))
    :effect (and (not (contains ?r ?o)) (inhand ?o))
  )

  (:action PutObject
    :parameters (?o - object ?r - receptacle)
    :precondition (and (at ?r) (inhand ?o))
    :effect (and (contains ?r ?o) (not (inhand ?o)))
  )

  (:action useObject
    :parameters (?o - object)
    :effect (used ?o)
  )

  (:action HeatObject
    :parameters (?o - object ?r - microwaveReceptacle)
    :precondition (and (at ?r) (inhand ?o))
    :effect (heated ?o)
  )

  (:action CleanObject
    :parameters (?o - object ?r - sinkbasinReceptacle)
    :precondition (and (at ?r) (inhand ?o))
    :effect (cleaned ?o)
  )

  (:action CoolObject
    :parameters (?o - object ?r - fridgeReceptacle)
    :precondition (and (at ?r) (inhand ?o))
    :effect (cooled ?o)
  )

  (:action SliceObject
    :parameters (?r - receptacle ?co - object ?sharp_o - sharpObject)
    :precondition (at ?r)
    :effect (sliced ?co)
  )
)
"""

pf = """(define (problem room_task_prob)
  (:domain room_task)
  (:objects 
    init_receptacle - receptacle
    cabinet20 cabinet19 cabinet18 cabinet17 cabinet16 cabinet15 cabinet14 cabinet13 cabinet12 cabinet11 cabinet10 cabinet9 cabinet8 cabinet7 cabinet6 cabinet5 cabinet4 cabinet3 cabinet2 cabinet1 - receptacle
    drawer6 drawer5 drawer4 drawer3 drawer2 drawer1 - receptacle
    coffeemachine1 countertop2 countertop1 diningtable2 diningtable1 garbagecan1 stoveburner4 stoveburner3 stoveburner2 stoveburner1 toaster1 - receptacle
    fridge1 - fridgeReceptacle
    microwave1 - microwaveReceptacle
    sinkbasin1 - sinkbasinReceptacle
    kettle3 mug1 plate1 glassbottle1 winebottle1 spoon2 spoon3 knife1 - object
    bowl2 bread1 butterknife1 dishsponge1 kettle1 spatula2 spatula1 tomato1 - object
  )
  (:init
    (at diningtable1)
    (visited cabinet20)
    (visited cabinet19)
    (visited cabinet18)
    (visited cabinet17)
    (visited cabinet16)
    (visited cabinet15)
    (visited cabinet14)
    (visited cabinet13)
    (visited cabinet12)
    (visited cabinet11)
    (visited cabinet10)
    (visited cabinet9)
    (visited cabinet8)
    (visited cabinet7)
    (visited cabinet6)
    (visited cabinet5)
    (visited cabinet4)
    (visited cabinet3)
    (visited cabinet2)
    (visited cabinet1)
    (visited drawer6)
    (visited drawer5)
    (visited drawer4)
    (visited drawer3)
    (visited drawer2)
    (visited drawer1)
    (visited diningtable1)

    (opened cabinet17) ; contains kettle3
    (opened cabinet15) ; opened but empty
    (opened cabinet10) ; opened but empty
    (opened cabinet9)  ; contains glassbottle1
    (opened cabinet13) ; opened but empty
    (opened cabinet8)  ; opened but empty
    (opened cabinet5)  ; opened but empty

    (contains cabinet17 kettle3)
    (contains cabinet12 plate1)
    (contains cabinet9 glassbottle1)
    (contains cabinet8 mug1)
    (contains drawer4 spoon2)
    (contains drawer5 spoon3)

    (contains diningtable1 bowl2)
    (contains diningtable1 bread1)
    (contains diningtable1 butterknife1)
    (contains diningtable1 dishsponge1)
    (contains diningtable1 kettle1)
    (contains diningtable1 spatula2)
    (contains diningtable1 spatula1)
    (contains diningtable1 tomato1)
    (contains diningtable1 winebottle1)
  )
  (:goal (and (cooled winebottle1) (contains diningtable1 winebottle1)))
)
"""

print(run_solver(df, pf, "dual-bfws-ffparser"))
