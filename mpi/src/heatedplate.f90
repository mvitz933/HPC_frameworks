program heatedplate
  use mpi

  implicit none

  integer, parameter:: maxiter=100000000
  integer           :: numargs, i, j, iterations = 0
  double precision  :: eps, mean, diff
  character(len=20) :: arg, filename
  double precision, allocatable, dimension(:,:)  :: u, w
  integer           :: N=500
  real              :: start, end, eltime

  interface
    subroutine set_bcs(rank,npes,nr,nc,u)
       implicit none
       integer, intent(in)                             :: rank, npes, nr, nc
       double precision, dimension(0:nr+1,0:nc+1), intent(inout) :: u
    end subroutine
  end interface

  ! check number of parameters and read in epsilon
  numargs=command_argument_count()
  if (numargs .ne. 2) then
     print *, 'USAGE: <epsilon> <output-file>'
     stop
  else
     call get_command_argument(1,arg)
     read(arg,*) eps
     call get_command_argument(2,filename)
  endif
  
  write (*,'(a,es15.3)') 'running until the difference is <= ', eps
 
  ! Usually we'd read in N from the command line or something, we set it 
  ! explicitly in this code for simplicity.

  allocate(u(0:N+1,0:N+1),w(0:N+1,0:N+1))

  ! Set boundary values and compute mean boundary value. 
  ! This has the ice bath on the top edge.
  ! Note: when plotting, 0,0 is the bottom.
  mean=sum(u(:,0))+sum(u(:,N+1))+sum(u(N+1,:))+sum(u(0,:))
  mean = mean / (4.0 * (N))

  u(1:N,1:N)=mean

  call set_bcs(rank,npes,N,N,u)

  diff=eps
  call cpu_time(start)

  ! Compute steady-state solution

  do while ( iterations<maxiter )
     do j=1,ncl
        do i=1,nr
           w(i,j) = 0.25*(u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1))
        enddo
     enddo
     diff=maxval(abs(w(1:N,1:N)-u(1:N,1:N)))

     if (diff <= eps) then
        exit
     endif

     u = w

     call set_bcs(rank,npes,nr,ncl,u)

     iterations = iterations + 1
  enddo


  call cpu_time(end)
  eltime = end-start

  write (*,*) 'completed; iterations = ', iterations
  write (*,*) 'Elapsed time is ', eltime


  ! Write solution to output file

  write(fname,'(a)') filename(1:len_trim(filename))

  open (unit=10,file=fname)

  do j=1,N
     write (10,*) u(1:N,j)
  enddo

  ! All done!
  write "wrote output file ", fname

end program heatedplate

subroutine set_bcs(rank,npes,nr,nc,u)
implicit none
integer, intent(in)                             :: rank, npes, nr, nc
double precision, dimension(0:nr+1,0:nc+1), intent(inout) :: u
double precision                                :: topBC, bottomBC, edgeBC

  ! Set boundary values and compute mean boundary value. 
  ! This has the ice bath on the top edge.
  ! Note: when plotting, 0,0 is the bottom.

  topBC=100.d0
  bottomBC=0.d0
  edgeBC=100.d0

  !Set boundary conditions
  if (rank==0) then
     u(:,0)=edgeBC
  else if (rank==npes-1) then
     u(:,nc+1)=edgeBC
  endif
  u(nr+1,:)=topBC
  u(0,:)=bottomBC 

end subroutine set_bcs

