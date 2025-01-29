(define (problem explore-house)
    (:domain exploration)
    (:objects 
        kitchen - location
        patio - location
        pantry - location
        backyard - location
        driveway - location
        street - location
        east - direction
        south - direction
        west - direction
        north - direction
    )
    (:init 
        (at backyard)
        (connected kitchen patio south)
        (connected kitchen pantry west)
        (connected pantry kitchen east)
        (connected backyard kitchen north)
        (connected backyard driveway south)
        (connected backyard street east)
        (connected backyard patio west)
        (door-open kitchen backyard)
        (door-open backyard kitchen)
        (door-closed backyard patio)
    )
    (:goal 
        (at driveway)
    )
)