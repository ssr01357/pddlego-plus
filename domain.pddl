(define (domain explore) 
(:types location direction)
(:predicates
    (at ?loc - location)
    (door ?loc1 - location ?loc2 - location ?dir - direction)
    (door_open ?loc1 - location ?loc2 - location ?dir - direction)
)
(:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and (door ?loc1 ?loc2 ?dir) (at ?loc1))
    :effect (door_open ?loc1 ?loc2 ?dir)
)
(:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (door_open ?from ?to ?dir))
    :effect (and (not (at ?from)) (at ?to))
)
)