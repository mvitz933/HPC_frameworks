program heatedplate
  implicit none

  integer, parameter:: N=500
  integer :: numargs, i, j, iterations = 0
  double precision :: eps, mean, diff = 1.0
  character(len=80) :: arg, filename
  double precision, dimension(0:N+1,0:N+1) :: u, w

  ! check number of parameters and read in epsilon
  numargs=iargc()
  if (numargs .ne. 2) then
     stop 'USAGE: <epsilon> <output-file>'
  endif
  call getarg(1, arg)
  read(arg,*) eps
  call getarg(2, filename)
  write (*,*) 'running until the difference is <= ', eps

  ! Set boundary values and compute mean boundary value. 
  ! This has the ice bath on the top edge, not the bottom edge.

  u=0.d0
  u(:,0)=100.d0
  u(:,N+1)=100.d0
  u(N+1,:)=100.d0


  ! Initialize interior values to the boundary mean
  mean=sum(u(:,0))+sum(u(:,N+1))+sum(u(N+1,:))
  mean = mean / (4.0 * N)
  print *, 'Mean ', mean

  u(1:N,1:N)=mean

  diff=eps

  w=u

  ! Compute steady-state solution
  do while ( diff >= eps )
     do j=1,N
        do i=1,N
           w(i,j) = (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1))/4.0
        enddo
     enddo
     diff=maxval(abs(w-u))
     if (diff >= eps) then
        u=w
     endif
     iterations = iterations + 1
  enddo
  write (*,*) 'completed; iterations = ', iterations

  ! Write solution to output file
  open (unit=10,file=filename)
  write (10,*) N
  do i=1,N
     do j=1,N
        write (10,*) u(i,j)
     enddo
  enddo
  close (10)

  ! All done!
  write (*,*) "wrote output file ", filename

end program heatedplate
