! $Id$
!
!  Energy-calibrated buried impact source for cratering runs.
!  ------------------------------------------------------------------
!  This initial condition is ADDITIVE: it is applied after the standard
!  init_uu / init_lnrho / init_ss (e.g. a 'zjump' density + entropy
!  interface that represents the planet surface) and before init_eos.
!
!  It deposits, in a compact Gaussian region buried a little below the
!  surface, an amount of energy E derived from the transient crater
!  radius you want (gravity-regime pi-scaling):
!
!      R_transient = crater_K * ( E / (rho_target * g) )**pexp
!      pexp = 1/3   (2-D, line source)   or   1/4   (3-D, point source)
!  =>  E = rho_target * g * (R_transient/crater_K)**(1/pexp)
!
!  A fraction  frac_kin  of E goes into a downward velocity kick
!  (u_z < 0), the remainder into internal energy, applied as an
!  entropy increment  d(ss) = cv * ln(T_new/T_old)  at fixed density.
!  Both parts are normalised with a global (MPI) integral of the
!  profile, so the deposited energy equals E to round-off.
!
!  Set  crater_energy > 0  to bypass the scaling and deposit a fixed energy.
!
!  ALL parameters below are in CODE UNITS, so this module does not care
!  whether the run uses  unit_system='cgs'  or  'SI'.  The startup printout
!  also reports the deposited energy in physical units, labelled from
!  unit_system (erg for cgs, J for SI).
!
!  Namelist  &initial_condition_pars :
!    crater_radius_target  wanted transient radius            [code length]
!    crater_K              pi-scaling prefactor (tune once)   [-]   (~1.1-1.5)
!    crater_g              |gravz|, MUST match grav_run_pars  [code accel]
!    rho_target            target (surface) density           [code dens]
!    crater_energy         >0 overrides the scaling           [code energy]
!    impact_x, impact_y    impact point (default: x-centre,0)  [code length]
!    impact_z              surface height (default: z-centre)  [code length]
!    impact_depth_sigma    burial depth in units of sigma      [-]
!    crater_sigma          deposition Gaussian radius          [code length]
!    frac_kin              kinetic fraction of E (0..1)        [-]
!    lhemisphere           deposit only where z <= impact_z    [-]
!    lset_impact           master on/off switch                [-]
!
!** AUTOMATIC CPARAM.INC GENERATION ****************************
! Declare (for generation of cparam.inc) the number of f array
! variables and auxiliary variables added by this module
!
! CPARAM logical, parameter :: linitial_condition = .true.
!
!***************************************************************
module InitialCondition
!
  use Cparam
  use Cdata
  use General, only: keep_compiler_quiet
  use Messages
!
  implicit none
!
  include '../initial_condition.h'
!
  real :: crater_radius_target = 0.15
  real :: crater_K             = 1.3
  real :: crater_g             = impossible
  real :: rho_target           = impossible
  real :: crater_energy        = 0.0
  real :: impact_x             = impossible
  real :: impact_y             = 0.0
  real :: impact_z             = impossible
  real :: impact_depth_sigma   = 1.0
  real :: crater_sigma         = 0.0
  real :: frac_kin             = 0.5
  logical :: lhemisphere       = .true.
  logical :: lset_impact       = .true.
!
  namelist /initial_condition_pars/ &
      crater_radius_target, crater_K, crater_g, rho_target, crater_energy, &
      impact_x, impact_y, impact_z, impact_depth_sigma, crater_sigma, &
      frac_kin, lhemisphere, lset_impact
!
  contains
!***********************************************************************
    subroutine register_initial_condition()
!
!  Register variables associated with this module; nothing to do here.
!
      if (lroot) call svn_id( &
          "$Id$")
!
    endsubroutine register_initial_condition
!***********************************************************************
    subroutine initialize_initial_condition(f)
!
      real, dimension (mx,my,mz,mfarray) :: f
!
      call keep_compiler_quiet(f)
!
    endsubroutine initialize_initial_condition
!***********************************************************************
    subroutine initial_condition_all(f,profiles)
!
!  Deposit the buried impact pulse on top of the existing f array.
!
      use EquationOfState, only: eoscalc, ilnrho_ss, get_gamma_etc
      use Mpicomm, only: mpiallreduce_sum, mpiallreduce_max
!
      real, dimension (mx,my,mz,mfarray), optional, intent(inout) :: f
      real, dimension (:,:),              optional, intent(out)   :: profiles
!
      real :: gamma, cp, cv
      real :: sig, xc, yc, zc, zsurf, gg, rhoT, pexp
      real :: Edep, Edep_phys, Ath, U0
      character(len=8) :: elabel
      real :: rr2, prof, dv, rho, lnTT_old, TT_old, dEthm, TT_new
      real :: Sth_loc, Sth, Skin_loc, Skin
      real :: Eth_loc, Eth, Ekin_loc, Ekin, TTmax_loc, TTmax
      integer :: l, m, n
!
      if (present(profiles)) call fatal_error('initial_condition_all', &
          'impact_crater does not implement the profiles argument')
      if (.not.present(f)) return
      if (.not.lset_impact) return
!
      if (ldensity_nolog) call fatal_error('initial_condition_all', &
          'impact_crater assumes logarithmic density (ldensity_nolog=F)')
      if (.not.lentropy) call fatal_error('initial_condition_all', &
          'impact_crater needs the entropy module')
      if (.not.lhydro) call fatal_error('initial_condition_all', &
          'impact_crater needs the hydro module')
      if (frac_kin<0.0 .or. frac_kin>1.0) call fatal_error('initial_condition_all', &
          'frac_kin must be in [0,1]')
!
      call get_gamma_etc(gamma,cp,cv)
!
!  Geometry (fill in defaults).
!
      xc = impact_x
      if (xc==impossible) xc = 0.5*(xyz0(1)+xyz1(1))
      yc = impact_y
      zsurf = impact_z
      if (zsurf==impossible) zsurf = 0.5*(xyz0(3)+xyz1(3))
!
      sig = crater_sigma
      if (sig<=0.0) sig = max(0.02, 6.0*dxmax)
      zc = zsurf - impact_depth_sigma*sig
!
      if (sig < 3.0*dxmax .and. lroot) call warning('impact_crater', &
          'crater_sigma < 3*dxmax: the deposition region is under-resolved')
!
!  Gravity and target density.
!
      gg = crater_g
      rhoT = rho_target
      if (rhoT==impossible) then
        rhoT = 1.0
        if (lroot) call warning('impact_crater', &
            'rho_target not set; assuming 1.0 - set it to your surface density')
      endif
!
!  Deposition energy in code units (everything here is unit-system-agnostic).
!
      if (crater_energy>0.0) then
        Edep = crater_energy
      else
        if (gg==impossible .or. gg<=0.0) call fatal_error('initial_condition_all', &
            'set crater_g (=|gravz|) for the crater-radius scaling, '// &
            'or give crater_energy directly')
        pexp = 1.0/(real(max(dimensionality,2))+1.0)   ! 1/3 in 2-D, 1/4 in 3-D
        Edep = rhoT*gg*(crater_radius_target/crater_K)**(1.0/pexp)
      endif
      Edep_phys = Edep*unit_energy                     ! erg (cgs) or J (SI)
!
!  Pass 1: global normalisation integrals.
!
      Sth_loc = 0.0 ; Skin_loc = 0.0
      do n=n1,n2 ; do m=m1,m2 ; do l=l1,l2
        if (lhemisphere .and. z(n) > zsurf) cycle
        rr2 = (x(l)-xc)**2 + (y(m)-yc)**2 + (z(n)-zc)**2
        prof = exp(-0.5*rr2/sig**2)
        if (prof < 1.0e-6) cycle
        dv  = dVol_x(l)*dVol_y(m)*dVol_z(n)
        rho = exp(f(l,m,n,ilnrho))
        Sth_loc  = Sth_loc  + prof*dv
        Skin_loc = Skin_loc + 0.5*rho*prof**2*dv
      enddo ; enddo ; enddo
      call mpiallreduce_sum(Sth_loc,Sth)
      call mpiallreduce_sum(Skin_loc,Skin)
!
      if (Sth<=0.0) call fatal_error('initial_condition_all', &
          'deposition region is empty - check impact_x/impact_z/crater_sigma')
!
      Ath = (1.0-frac_kin)*Edep/Sth
      U0  = 0.0
      if (frac_kin>0.0 .and. Skin>0.0) U0 = sqrt(frac_kin*Edep/Skin)
!
!  Pass 2: apply the entropy and velocity increments.
!
      Eth_loc = 0.0 ; Ekin_loc = 0.0 ; TTmax_loc = 0.0
      do n=n1,n2 ; do m=m1,m2 ; do l=l1,l2
        if (lhemisphere .and. z(n) > zsurf) cycle
        rr2 = (x(l)-xc)**2 + (y(m)-yc)**2 + (z(n)-zc)**2
        prof = exp(-0.5*rr2/sig**2)
        if (prof < 1.0e-6) cycle
        dv  = dVol_x(l)*dVol_y(m)*dVol_z(n)
        rho = exp(f(l,m,n,ilnrho))
        call eoscalc(ilnrho_ss,f(l,m,n,ilnrho),f(l,m,n,iss),lnTT=lnTT_old)
        TT_old = exp(lnTT_old)
        dEthm  = Ath*prof/rho                 ! internal energy per unit mass
        TT_new = TT_old + dEthm/cv
        f(l,m,n,iss) = f(l,m,n,iss) + cv*log(TT_new/TT_old)
        f(l,m,n,iuz) = f(l,m,n,iuz) - U0*prof
        Eth_loc  = Eth_loc  + rho*cv*(TT_new-TT_old)*dv
        Ekin_loc = Ekin_loc + 0.5*rho*(U0*prof)**2*dv
        TTmax_loc = max(TTmax_loc,TT_new)
      enddo ; enddo ; enddo
      call mpiallreduce_sum(Eth_loc,Eth)
      call mpiallreduce_sum(Ekin_loc,Ekin)
      call mpiallreduce_max(TTmax_loc,TTmax)
!
      if (lroot) then
        if (unit_system=='SI') then
          elabel = 'J'
        elseif (unit_system=='cgs') then
          elabel = 'erg'
        else
          elabel = '?'
        endif
        if (dimensionality<3) elabel = trim(elabel)//'/L_y'
        print*,'------------- impact_crater initial condition -------------'
        print*,'  gamma, cp, cv             = ',gamma,cp,cv
        print*,'  impact (x,y), surface z   = ',xc,yc,zsurf
        print*,'  blob centre z, sigma      = ',zc,sig
        print*,'  dxmax                     = ',dxmax
        print*,'  |g|, rho_target           = ',gg,rhoT
        print*,'  dimensionality            = ',dimensionality
        print*,'  E deposited  [code]       = ',Edep
        print*,'  E deposited  ['//trim(elabel)//'] = ',Edep_phys
        print*,'  frac_kin, peak |u_z| (U0) = ',frac_kin,U0
        print*,'  achieved E_thermal [code] = ',Eth
        print*,'  achieved E_kinetic [code] = ',Ekin
        print*,'  peak T in blob     [code] = ',TTmax
        print*,'  (calibrate crater_K against the measured transient radius)'
        print*,'----------------------------------------------------------'
      endif
!
    endsubroutine initial_condition_all
!***********************************************************************
    subroutine initial_condition_uu(f)
      real, dimension (mx,my,mz,mfarray), intent(inout) :: f
      call keep_compiler_quiet(f)
    endsubroutine initial_condition_uu
!***********************************************************************
    subroutine initial_condition_lnrho(f)
      real, dimension (mx,my,mz,mfarray), intent(inout) :: f
      call keep_compiler_quiet(f)
    endsubroutine initial_condition_lnrho
!***********************************************************************
    subroutine initial_condition_ss(f)
      real, dimension (mx,my,mz,mfarray), intent(inout) :: f
      call keep_compiler_quiet(f)
    endsubroutine initial_condition_ss
!***********************************************************************
    subroutine read_initial_condition_pars(iostat)
!
      use File_io, only: parallel_unit
!
      integer, intent(out) :: iostat
!
      read(parallel_unit, NML=initial_condition_pars, IOSTAT=iostat)
!
    endsubroutine read_initial_condition_pars
!***********************************************************************
    subroutine write_initial_condition_pars(unit)
!
      integer, intent(in) :: unit
!
      write(unit, NML=initial_condition_pars)
!
    endsubroutine write_initial_condition_pars
!***********************************************************************
!********************************************************************
!************        DO NOT DELETE THE FOLLOWING       **************
!********************************************************************
!**  This is an automatically generated include file that creates  **
!**  copies dummy routines from noinitial_condition.f90 for any     **
!**  InitialCondition routines not implemented in this file         **
!**                                                                 **
    include '../initial_condition_dummies.inc'
!********************************************************************
endmodule InitialCondition
