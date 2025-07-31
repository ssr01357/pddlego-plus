(define (problem exploration-problem)
  (:domain exploration)
  
  (:objects
    kitchen patio west_room - location
    south west - direction
  )
  
  (:init
    (at kitchen)
    (door-closed kitchen patio south)
    (door-closed kitchen west_room west)
  )
  
  (:goal
    (at west_room)
  )
)