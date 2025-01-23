(define (problem explore-environment)
(:domain explore)
(:objects
    kitchen - location
    patio - location
    backyard - location
    driveway - location
    street - location
    south - direction
    north - direction
    east - direction
    west - direction
)
(:init
    (at driveway)
    (door kitchen patio south)
    (door patio backyard south)
    (door backyard street east)
    (door driveway backyard north)
    (door_open kitchen patio south)
    (door_open patio backyard south)
    (door_open driveway backyard north)
)
(:goal
    (exists (?loc - location) (and (not (at kitchen)) (not (at patio)) (not (at backyard)) (not (at driveway))))
)
)